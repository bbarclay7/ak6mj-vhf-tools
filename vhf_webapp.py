#!/usr/bin/env -S uv run
# -*- mode: python; -*-
# vim: set ft=python:
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "flask",
#   "numpy",
#   "matplotlib",
#   "srtm.py",
#   "requests",
# ]
# ///
"""
VHF Path Analysis Web App
Flask-based GUI for VHF path analysis and link budget calculations

Run with: uv run vhf_webapp.py
Access at: http://localhost:5001 or behind proxy at https://www.shoeph.one/vhf/
"""

from functools import wraps
from flask import Flask, Blueprint, render_template, request, jsonify, send_file, Response
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import json
import math
import io
import base64
from pathlib import Path
from datetime import datetime
import requests
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

try:
    import srtm
    SRTM_AVAILABLE = True
except ImportError:
    SRTM_AVAILABLE = False

# Determine if running behind proxy at /vhf/
# URL_PREFIX is used on the blueprint so Flask generates correct URLs
URL_PREFIX = os.getenv('URL_PREFIX', '')

# Basic auth for write operations (override via environment if needed)
AUTH_USERNAME = os.getenv('VHF_AUTH_USER', 'admin')
AUTH_PASSWORD = os.getenv('VHF_AUTH_PASS', '73palomar')


def requires_auth(f):
    """Require HTTP Basic Auth for this endpoint (only if AUTH_PASSWORD is set)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_PASSWORD:
            return f(*args, **kwargs)  # Auth disabled

        auth = request.authorization
        if not auth or auth.username != AUTH_USERNAME or auth.password != AUTH_PASSWORD:
            return Response(
                'Authentication required for this operation.',
                401,
                {'WWW-Authenticate': 'Basic realm="VHF Tools"'}
            )
        return f(*args, **kwargs)
    return decorated


app = Flask(__name__)

# Support running behind reverse proxy
if URL_PREFIX:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Create blueprint with URL prefix for correct URL generation
bp = Blueprint('main', __name__, url_prefix=URL_PREFIX)

# Data directories - production vs local
PRODUCTION_DATA_DIR = Path('/var/www/local/vhf-data')
if PRODUCTION_DATA_DIR.exists() or os.getenv('VHF_PRODUCTION'):
    DATA_DIR = PRODUCTION_DATA_DIR
    SRTM_CACHE_DIR = PRODUCTION_DATA_DIR / 'srtm'
else:
    DATA_DIR = Path.home() / '.cache' / 'vhf-tools'
    SRTM_CACHE_DIR = Path.home() / '.cache' / 'srtm'

CACHE_FILE = DATA_DIR / 'locations.json'
SETTINGS_FILE = DATA_DIR / 'settings.json'
ANTENNAS_FILE = DATA_DIR / 'antennas.json'
PLOT_DIR = Path('static/plots')
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configure SRTM cache directory
if SRTM_AVAILABLE:
    SRTM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Set SRTM cache path via environment variable (srtm.py reads this)
    os.environ['SRTM1_DIR'] = str(SRTM_CACHE_DIR)
    os.environ['SRTM3_DIR'] = str(SRTM_CACHE_DIR)

# Node aliases
NODE_ALIASES = {
    'kberr': {'callsign': 'KE6YUV', 'name': 'Berryessa Peak Digipeater',
              'lat': 38.7833, 'lon': -122.1167, 'elevation': 914},  # 3000ft peak
    'kjohn': {'callsign': 'KF6ANX-4', 'name': 'Livermore Node',
              'lat': 37.6819, 'lon': -121.7681, 'elevation': 457},  # On Mt. Diablo area peak
    'cool': {'callsign': 'KM6LYW-4', 'name': 'Cool BBS Node',
             'lat': 38.8894, 'lon': -121.0156, 'elevation': None},  # Ground level
    'waqth': {'callsign': 'AK6MJ', 'name': 'WA QTH - Freeland',
              'lat': 48.0356948, 'lon': -122.5419147, 'elevation': None},  # Whidbey Island
}

# Default antenna definitions - gain values in dBi, typical heights in meters
# Sources: W8JI measurements, Diamond mfr specs, ARRL Antenna Book
DEFAULT_ANTENNAS = {
    'rubber_ducky': {'name': 'Rubber Ducky (HT)', 'gain_dbi': -1.0, 'default_height': 1.5},
    'signal_stick': {'name': 'SignalStuff Signal Stick', 'gain_dbi': 2.0, 'default_height': 1.5},
    'dipole': {'name': 'Dipole', 'gain_dbi': 2.15, 'default_height': 5.0},
    'groundplane': {'name': 'Ground Plane (1/4 wave)', 'gain_dbi': 1.5, 'default_height': 5.0},
    'j_pole': {'name': 'J-Pole', 'gain_dbi': 2.5, 'default_height': 5.0},
    'slim_jim': {'name': 'Slim Jim', 'gain_dbi': 2.5, 'default_height': 5.0},
    'moxon': {'name': 'Moxon', 'gain_dbi': 6.0, 'default_height': 8.0},
    'diamond_x50': {'name': 'Diamond X50', 'gain_dbi': 4.5, 'default_height': 8.0},
    'diamond_x200': {'name': 'Diamond X200', 'gain_dbi': 6.0, 'default_height': 10.0},
    'diamond_x300': {'name': 'Diamond X300', 'gain_dbi': 6.5, 'default_height': 10.0},
    'yagi_3el': {'name': '3-element Yagi', 'gain_dbi': 7.0, 'default_height': 10.0},
    'yagi_5el': {'name': '5-element Yagi', 'gain_dbi': 9.5, 'default_height': 12.0},
    'yagi_7el': {'name': '7-element Yagi', 'gain_dbi': 11.5, 'default_height': 15.0},
}

def load_antennas():
    """Load antennas from config file or use defaults"""
    if ANTENNAS_FILE.exists():
        try:
            with open(ANTENNAS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_ANTENNAS.copy()

def save_antennas(antennas):
    """Save antennas to config file"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANTENNAS_FILE, 'w') as f:
        json.dump(antennas, f, indent=2)

def load_settings():
    """Load combined settings for API compatibility"""
    return {
        'node_aliases': NODE_ALIASES.copy(),
        'antennas': load_antennas()
    }

def save_settings(settings):
    """Save settings - splits into separate files"""
    if 'antennas' in settings:
        save_antennas(settings['antennas'])
    # Node aliases are now in settings file for backwards compat
    if 'node_aliases' in settings:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({'node_aliases': settings['node_aliases']}, f, indent=2)

def load_cache():
    """Load location cache"""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    """Save location cache"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def lookup_callsign(callsign):
    """Look up callsign from FCC database"""
    import requests
    callsign = callsign.upper().strip()
    url = f"https://callook.info/{callsign}/json"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'VALID':
                location = data.get('location', {})
                lat = location.get('latitude')
                lon = location.get('longitude')
                grid = location.get('gridsquare')
                name = data.get('name', '')
                
                if lat and lon:
                    return {
                        'lat': float(lat),
                        'lon': float(lon),
                        'grid': grid,
                        'name': name,
                        'callsign': callsign,
                        'type': 'callsign'
                    }
    except:
        pass
    return None

def geocode_address(address):
    """Geocode an address using Nominatim (OSM) with improved search strategies"""

    headers = {
        'User-Agent': 'VHF-Path-Analysis-AK6MJ/1.0 (Amateur Radio Tools)'
    }

    # Try Nominatim with original query
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json',
        'limit': 1,
        'addressdetails': 1
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.ok:
            results = response.json()
            if results and len(results) > 0:
                result = results[0]
                print(f"Nominatim geocoded '{address}' -> {result['lat']}, {result['lon']}", flush=True)
                return {
                    'lat': float(result['lat']),
                    'lon': float(result['lon']),
                    'name': result.get('display_name', address),
                    'type': 'address',
                    'source': 'nominatim'
                }
    except Exception as e:
        print(f"Nominatim geocoding error for '{address}': {e}", flush=True)

    # Try with structured query if it looks like a US address
    # Extract street, city, state if possible
    if ',' in address:
        parts = [p.strip() for p in address.split(',')]
        if len(parts) >= 2:
            # Try structured search: street, city state
            structured_params = {
                'format': 'json',
                'limit': 1,
                'addressdetails': 1,
                'countrycodes': 'us'
            }

            # Heuristic: if last part looks like state code (2 chars) or state name
            if len(parts[-1]) <= 2:  # Likely state abbreviation
                if len(parts) == 3:  # street, city, state
                    structured_params['street'] = parts[0]
                    structured_params['city'] = parts[1]
                    structured_params['state'] = parts[2]
                elif len(parts) == 2:  # city, state
                    structured_params['city'] = parts[0]
                    structured_params['state'] = parts[1]

            if len(structured_params) > 4:  # Has more than base params
                try:
                    response = requests.get(url, params=structured_params, headers=headers, timeout=10)
                    if response.ok:
                        results = response.json()
                        if results and len(results) > 0:
                            result = results[0]
                            print(f"Nominatim structured search '{address}' -> {result['lat']}, {result['lon']}", flush=True)
                            return {
                                'lat': float(result['lat']),
                                'lon': float(result['lon']),
                                'name': result.get('display_name', address),
                                'type': 'address',
                                'source': 'nominatim-structured'
                            }
                except Exception as e:
                    print(f"Nominatim structured search error: {e}", flush=True)

    print(f"Could not geocode address: '{address}'", flush=True)
    return None

def maidenhead_to_latlon(grid):
    """Convert grid square to lat/lon"""
    grid = grid.upper()
    if len(grid) < 4:
        raise ValueError("Grid too short")

    lon = (ord(grid[0]) - ord('A')) * 20 - 180
    lat = (ord(grid[1]) - ord('A')) * 10 - 90
    lon += int(grid[2]) * 2
    lat += int(grid[3]) * 1

    if len(grid) >= 6:
        lon += (ord(grid[4]) - ord('A')) * (2/24) + (1/24)
        lat += (ord(grid[5]) - ord('A')) * (1/24) + (1/48)
    else:
        lon += 1
        lat += 0.5

    return lat, lon

def parse_location(location_str, cache):
    """Parse location string"""
    location_str = location_str.strip()
    location_key = location_str.lower()
    
    # Check cache
    if location_key in cache:
        return cache[location_key]
    
    # Check node aliases
    if location_key in NODE_ALIASES:
        node = NODE_ALIASES[location_key]
        if 'lat' in node and 'lon' in node:
            loc = {
                'lat': node['lat'],
                'lon': node['lon'],
                'label': location_str.upper(),
                'name': node['name'],
                'type': 'node_alias'
            }
            cache[location_key] = loc
            save_cache(cache)
            return loc
    
    # Check if coordinates
    if ',' in location_str:
        try:
            lat, lon = map(float, location_str.split(','))
            return {'lat': lat, 'lon': lon, 'label': location_str, 'type': 'coordinates'}
        except:
            pass
    
    # Check if grid square
    if len(location_str) in [4, 6, 8]:
        try:
            lat, lon = maidenhead_to_latlon(location_str)
            return {'lat': lat, 'lon': lon, 'grid': location_str.upper(), 
                   'label': location_str.upper(), 'type': 'grid'}
        except:
            pass
    
    # Try callsign lookup
    loc = lookup_callsign(location_str)
    if loc:
        cache[location_key] = loc
        save_cache(cache)
        return loc

    # Try geocoding as final fallback (for addresses, cities, landmarks)
    loc = geocode_address(location_str)
    if loc:
        loc['label'] = location_str
        cache[location_key] = loc
        save_cache(cache)
        return loc

    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km"""
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def calculate_link_budget(distance_km, freq_mhz, tx_power_w, tx_antenna_gain_dbi,
                         rx_antenna_gain_dbi, tx_height_m, rx_height_m):
    """Calculate VHF link budget"""
    # Free space path loss
    fspl_db = 32.45 + 20 * math.log10(distance_km) + 20 * math.log10(freq_mhz)
    
    # TX power in dBm
    tx_power_dbm = 10 * math.log10(tx_power_w * 1000)
    
    # RX power
    rx_power_dbm = (tx_power_dbm + tx_antenna_gain_dbi + rx_antenna_gain_dbi - fspl_db)
    
    # Fade margin (typical receiver sensitivity: -118 dBm for 1200 baud packet)
    rx_sensitivity_dbm = -118
    fade_margin_db = rx_power_dbm - rx_sensitivity_dbm
    
    return {
        'fspl_db': fspl_db,
        'tx_power_dbm': tx_power_dbm,
        'rx_power_dbm': rx_power_dbm,
        'fade_margin_db': fade_margin_db,
        'viable': fade_margin_db > 0
    }

def generate_elevation_plot(tx_info, rx_info, tx_height_m, rx_height_m, freq_mhz, imperial=False):
    """Generate elevation profile plot with overhead terrain map and return as base64"""
    if not SRTM_AVAILABLE:
        return None, "SRTM library not available"
    
    try:
        tx_lat, tx_lon = tx_info['lat'], tx_info['lon']
        rx_lat, rx_lon = rx_info['lat'], rx_info['lon']
        
        # Unit conversion
        if imperial:
            dist_factor = 0.621371
            elev_factor = 3.28084
            dist_unit = 'mi'
            elev_unit = 'ft'
        else:
            dist_factor = 1.0
            elev_factor = 1.0
            dist_unit = 'km'
            elev_unit = 'm'
        
        # Get elevation data
        elevation_data = srtm.get_data()
        distance_km = haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
        
        # Sample elevations along path
        num_points = 100
        lats = np.linspace(tx_lat, rx_lat, num_points)
        lons = np.linspace(tx_lon, rx_lon, num_points)
        elevations = []
        
        for lat, lon in zip(lats, lons):
            try:
                elev = elevation_data.get_elevation(lat, lon)
                elevations.append(elev if elev is not None else 0)
            except:
                elevations.append(0)
        
        distances = np.linspace(0, distance_km, num_points)
        
        # Get endpoint elevations - use override if specified in location info
        if 'elevation' in tx_info and tx_info['elevation'] is not None:
            tx_ground = tx_info['elevation']
        else:
            tx_ground = elevation_data.get_elevation(tx_lat, tx_lon) or 0
            
        if 'elevation' in rx_info and rx_info['elevation'] is not None:
            rx_ground = rx_info['elevation']
        else:
            rx_ground = elevation_data.get_elevation(rx_lat, rx_lon) or 0
        
        tx_altitude = tx_ground + tx_height_m
        rx_altitude = rx_ground + rx_height_m
        
        # Calculate LOS with curvature
        k = 4/3
        earth_radius_km = 6371
        
        los_profile = []
        elevations_corrected = []
        
        for i, d in enumerate(distances):
            frac = d / distance_km
            los_alt = tx_altitude + frac * (rx_altitude - tx_altitude)
            los_profile.append(los_alt)
            
            curve = (d * (distance_km - d)) / (2 * k * earth_radius_km)
            elevations_corrected.append(elevations[i] + curve * 1000)
        
        # Calculate Fresnel zone
        wavelength_m = 299.792458 / freq_mhz
        fresnel_radii = []
        
        for d in distances:
            if d == 0 or d == distance_km:
                fresnel_radii.append(0)
            else:
                d1 = d
                d2 = distance_km - d
                fresnel = 0.6 * math.sqrt((wavelength_m * d1 * d2 * 1000) / (d1 + d2))
                fresnel_radii.append(fresnel)
        
        # Check clearance - evaluate obstruction relative to Fresnel zone
        # VHF can knife-edge diffract, so minor LOS obstruction isn't fatal
        clearances = np.array(los_profile) - np.array(elevations_corrected)
        min_clearance = np.min(clearances)
        worst_idx = np.argmin(clearances)

        fresnel_array = np.array(fresnel_radii)
        fresnel_at_worst = fresnel_array[worst_idx] if worst_idx < len(fresnel_array) else 10

        if min_clearance >= fresnel_at_worst:
            # Full Fresnel clearance - excellent
            is_clear = True
            obstruction_severity = 0
        elif min_clearance >= 0:
            # LOS clear but Fresnel partially obstructed - normal for ground VHF
            upper_fresnel_penetration = (fresnel_at_worst - min_clearance) / fresnel_at_worst
            obstruction_severity = upper_fresnel_penetration * 50  # Max 50% if LOS clear
            is_clear = True
        elif min_clearance > -fresnel_at_worst * 0.6:
            # Terrain slightly above LOS but within 60% Fresnel - knife-edge diffraction likely
            # Scale from 50% (at LOS) to 90% (at -0.6 Fresnel)
            penetration_ratio = abs(min_clearance) / (fresnel_at_worst * 0.6)
            obstruction_severity = 50 + penetration_ratio * 40
            is_clear = obstruction_severity < 70  # Marginal but may work
        else:
            # Severe obstruction - terrain well above LOS
            is_clear = False
            obstruction_severity = 90 + min(10, abs(min_clearance) / fresnel_at_worst * 10)
        
        # Calculate aspect ratio for overhead map to determine figure dimensions
        lat_range = abs(rx_lat - tx_lat) * 1.2  # Add margin
        lon_range = abs(rx_lon - tx_lon) * 1.2
        
        # For equal aspect ratio map, figure width should match lon:lat ratio
        # Base dimensions
        base_width = 14
        if lon_range > 0.001:  # Avoid division by zero for vertical paths
            map_aspect = lat_range / lon_range  # height/width of map data
            # Adjust figure to accommodate this aspect while filling width
            # Map gets 1/3 of height, so total height = map_height * 3
            # map_height = map_width * map_aspect
            # map_width fills figure width
            adjusted_height = base_width * map_aspect * 3
            # Clamp to reasonable limits for phone/desktop viewing
            # Max ~15 keeps it scrollable on phones, min 8 for very wide paths
            adjusted_height = max(8, min(adjusted_height, 15))
        else:
            # Nearly vertical path - use moderate height
            adjusted_height = 12
        
        # Create figure with subplot grid
        fig = plt.figure(figsize=(base_width, adjusted_height))
        ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1, rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0), colspan=1, rowspan=1)
        
        # === ELEVATION PROFILE ===
        # Convert to display units
        distances_display = distances * dist_factor
        elevations_display = np.array(elevations_corrected) * elev_factor
        los_display = np.array(los_profile) * elev_factor
        fresnel_upper = los_display + np.array(fresnel_radii) * elev_factor
        fresnel_lower = los_display - np.array(fresnel_radii) * elev_factor
        
        # Plot terrain
        ax1.fill_between(distances_display, 0, elevations_display, 
                        alpha=0.3, color='brown', label='Terrain')
        ax1.plot(distances_display, elevations_display, 'brown', linewidth=2)
        ax1.plot(distances_display, los_display, 'b--', linewidth=2, label='Line of Sight')
        ax1.fill_between(distances_display, fresnel_lower, fresnel_upper,
                        alpha=0.2, color='yellow', label='60% Fresnel Zone')
        
        # Mark stations
        tx_label = tx_info.get('label', tx_info.get('callsign', 'TX'))
        rx_label = rx_info.get('label', rx_info.get('callsign', 'RX'))
        
        ax1.plot(0, tx_altitude * elev_factor, 'go', markersize=10, label=f'{tx_label}')
        ax1.plot(distance_km * dist_factor, rx_altitude * elev_factor, 'ro', 
               markersize=10, label=f'{rx_label}')
        
        # Formatting
        ax1.set_xlabel(f'Distance ({dist_unit})', fontsize=11, fontweight='bold')
        ax1.set_ylabel(f'Elevation ({elev_unit})', fontsize=11, fontweight='bold')
        ax1.set_title(f'VHF Path Profile: {tx_label} to {rx_label}\n' +
                    f'Distance: {distance_km * dist_factor:.2f} {dist_unit} | ' +
                    f'Frequency: {freq_mhz} MHz',
                    fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Status box - graduated based on obstruction severity
        if obstruction_severity < 20:
            # Excellent clearance
            status = f"✓ CLEAR\n{min_clearance * elev_factor:.0f}{elev_unit} clearance"
            color = 'green'
        elif obstruction_severity < 50:
            # LOS clear, some Fresnel obstruction - typical for ground VHF
            status = f"✓ GOOD\n{min_clearance * elev_factor:.0f}{elev_unit} clearance"
            color = 'green'
        elif obstruction_severity < 70:
            # Minor LOS obstruction or significant Fresnel - marginal but workable
            if min_clearance < 0:
                status = f"⚠ MARGINAL\n{abs(min_clearance) * elev_factor:.0f}{elev_unit} over LOS"
            else:
                status = f"⚠ MARGINAL\nFresnel obstructed"
            color = 'orange'
        elif obstruction_severity < 90:
            # Significant obstruction - may work with diffraction
            status = f"⚠ OBSTRUCTED\n{abs(min_clearance) * elev_factor:.0f}{elev_unit} over LOS"
            color = 'orange'
        else:
            # Severe obstruction - unlikely to work
            status = f"✗ BLOCKED\n{abs(min_clearance) * elev_factor:.0f}{elev_unit} over LOS"
            color = 'red'
        
        ax1.text(0.5, 0.02, status, transform=ax1.transAxes, fontsize=10,
               fontweight='bold', verticalalignment='bottom', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
               color='white')
        
        # === OVERHEAD TERRAIN MAP ===
        # Calculate terrain grid - always square with margin
        margin = 0.15
        lat_range = abs(rx_lat - tx_lat)
        lon_range = abs(rx_lon - tx_lon)

        # Make square by expanding the smaller dimension
        # Account for longitude compression at latitude
        mid_lat = (tx_lat + rx_lat) / 2
        lon_scale = math.cos(math.radians(mid_lat))  # ~0.75 at 40°N

        # Convert to approximate equal distances
        lat_dist = lat_range
        lon_dist = lon_range * lon_scale

        # Use the larger dimension, add margin, then expand the smaller
        max_range = max(lat_dist, lon_dist) * (1 + 2 * margin)
        if max_range < 0.01:  # Minimum size for very close points
            max_range = 0.05

        # Center points
        center_lat = (tx_lat + rx_lat) / 2
        center_lon = (tx_lon + rx_lon) / 2

        # Calculate bounds as square (in distance terms)
        half_range = max_range / 2
        map_lat_min = center_lat - half_range
        map_lat_max = center_lat + half_range
        map_lon_min = center_lon - half_range / lon_scale
        map_lon_max = center_lon + half_range / lon_scale
        
        grid_points = 25
        grid_lats = np.linspace(map_lat_min, map_lat_max, grid_points)
        grid_lons = np.linspace(map_lon_min, map_lon_max, grid_points)
        terrain_grid = np.zeros((grid_points, grid_points))
        
        for i, lat in enumerate(grid_lats):
            for j, lon in enumerate(grid_lons):
                try:
                    elev = elevation_data.get_elevation(lat, lon)
                    terrain_grid[i, j] = (elev if elev is not None else 0) * elev_factor
                except:
                    terrain_grid[i, j] = 0
        
        # Plot terrain with gist_earth colormap
        contourf = ax2.contourf(grid_lons, grid_lats, terrain_grid, 
                                 levels=12, cmap='gist_earth', alpha=0.7)
        cbar = plt.colorbar(contourf, ax=ax2, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label(f'Elevation ({elev_unit})', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        # Add contour lines
        contour_lines = ax2.contour(grid_lons, grid_lats, terrain_grid,
                                     levels=6, colors='black', alpha=0.3, linewidths=0.5)
        
        # Plot path
        path_lats = np.linspace(tx_lat, rx_lat, 50)
        path_lons = np.linspace(tx_lon, rx_lon, 50)
        ax2.plot(path_lons, path_lats, 'b-', linewidth=3, 
                 path_effects=[path_effects.withStroke(linewidth=5, foreground='white')])
        
        # Plot stations
        ax2.plot(tx_lon, tx_lat, 'go', markersize=10, markeredgecolor='white', markeredgewidth=2)
        ax2.plot(rx_lon, rx_lat, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # Labels
        ax2.set_xlabel('Longitude', fontsize=9)
        ax2.set_ylabel('Latitude', fontsize=9)
        ax2.set_title('Overhead Terrain View', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_aspect('equal')
        ax2.tick_params(labelsize=8)
        
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_base64, None, {
            'is_clear': is_clear,
            'min_clearance': min_clearance,
            'obstruction_severity': obstruction_severity,
            'terrain_above_los': min_clearance < 0
        }
        
    except Exception as e:
        return None, str(e), None

@bp.route('/')
def index():
    """Main page"""
    antennas = load_antennas()
    return render_template('index.html',
                          antennas=antennas,
                          node_aliases=NODE_ALIASES,
                          url_prefix=URL_PREFIX)

@bp.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze path"""
    try:
        data = request.json
        
        tx_location = data.get('tx_location')
        rx_location = data.get('rx_location')
        tx_height = float(data.get('tx_height', 1.5))
        rx_height = float(data.get('rx_height', 1.5))
        tx_power = float(data.get('tx_power', 5))
        freq = float(data.get('freq', 145.050))
        tx_antenna = data.get('tx_antenna', 'dipole')
        rx_antenna = data.get('rx_antenna', 'dipole')
        imperial = data.get('imperial', False)
        generate_plot = data.get('generate_plot', True)
        
        # Get antenna gains from config
        antennas = load_antennas()
        tx_gain = antennas.get(tx_antenna, {}).get('gain_dbi', 2.15)
        rx_gain = antennas.get(rx_antenna, {}).get('gain_dbi', 2.15)
        
        cache = load_cache()
        
        # Parse locations
        tx_info = parse_location(tx_location, cache)
        if not tx_info:
            return jsonify({'error': f'Could not find location: {tx_location}'}), 400
        
        rx_info = parse_location(rx_location, cache)
        if not rx_info:
            return jsonify({'error': f'Could not find location: {rx_location}'}), 400
        
        # Calculate distance
        distance_km = haversine_distance(tx_info['lat'], tx_info['lon'], 
                                        rx_info['lat'], rx_info['lon'])
        
        # Calculate link budget
        link = calculate_link_budget(distance_km, freq, tx_power, tx_gain, rx_gain,
                                     tx_height, rx_height)
        
        # Generate plot if requested
        plot_base64 = None
        plot_error = None
        obstruction_info = None
        if generate_plot and SRTM_AVAILABLE:
            plot_base64, plot_error, obstruction_info = generate_elevation_plot(
                tx_info, rx_info, tx_height, rx_height, freq, imperial
            )
            
            # Adjust link budget for terrain obstruction
            if obstruction_info:
                if obstruction_info['terrain_above_los']:
                    # Terrain blocks LOS - check how severe
                    clearance_deficit = abs(obstruction_info['min_clearance'])
                    
                    if clearance_deficit > 100:  # More than 100m above LOS
                        # Extreme blockage - complete path obstruction
                        link['diffraction_loss_db'] = 60  # Very high loss (really: blocked)
                        link['rx_power_dbm'] -= 60
                        link['fade_margin_db'] -= 60
                        link['viable'] = False  # Force not viable for extreme blockage
                        link['obstruction_note'] = f'Path blocked - terrain {clearance_deficit:.0f}m above LOS (requires repeater/digipeater)'
                    elif clearance_deficit > 20:  # 20-100m above LOS
                        # Severe blockage - very limited propagation
                        link['diffraction_loss_db'] = 40
                        link['rx_power_dbm'] -= 40
                        link['fade_margin_db'] -= 40
                        link['viable'] = link['fade_margin_db'] > 10  # Need good margin to overcome
                        link['obstruction_note'] = f'Severe obstruction - terrain {clearance_deficit:.0f}m above LOS'
                    else:  # < 20m above LOS
                        # Moderate blockage - knife-edge diffraction possible
                        link['diffraction_loss_db'] = 20
                        link['rx_power_dbm'] -= 20
                        link['fade_margin_db'] -= 20
                        link['viable'] = link['fade_margin_db'] > 0
                        link['obstruction_note'] = f'Knife-edge diffraction - terrain {clearance_deficit:.0f}m above LOS'
                elif not obstruction_info['is_clear']:
                    # Penetrates upper Fresnel - moderate diffraction loss
                    severity = obstruction_info['obstruction_severity']
                    if severity > 75:
                        loss_db = 15  # Severe
                    elif severity > 50:
                        loss_db = 8   # Moderate
                    else:
                        loss_db = 3   # Minor
                    link['diffraction_loss_db'] = loss_db
                    link['rx_power_dbm'] -= loss_db
                    link['fade_margin_db'] -= loss_db
                    link['viable'] = link['fade_margin_db'] > 0
                    link['obstruction_note'] = f'{severity:.0f}% upper Fresnel obstruction'
                else:
                    link['diffraction_loss_db'] = 0
                    link['obstruction_note'] = 'Clear path'
        
        # Unit conversion
        if imperial:
            distance_display = distance_km * 0.621371
            dist_unit = 'mi'
        else:
            distance_display = distance_km
            dist_unit = 'km'
        
        return jsonify({
            'success': True,
            'tx_info': tx_info,
            'rx_info': rx_info,
            'distance': distance_display,
            'distance_unit': dist_unit,
            'link_budget': link,
            'plot': plot_base64,
            'plot_error': plot_error
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/locations')
def get_locations():
    """Get cached locations"""
    cache = load_cache()
    locations = []
    
    for key, loc in cache.items():
        locations.append({
            'key': key,
            'label': loc.get('label', key),
            'type': loc.get('type', 'unknown'),
            'lat': loc.get('lat'),
            'lon': loc.get('lon')
        })
    
    return jsonify({'locations': locations})

@bp.route('/api/antennas', methods=['GET'])
def get_antennas():
    """Get all antennas"""
    antennas = load_antennas()
    return jsonify(antennas)

@bp.route('/api/antennas', methods=['POST'])
@requires_auth
def add_antenna():
    """Add or update antenna"""
    try:
        data = request.json
        key = data.get('key', '').strip().lower().replace(' ', '_')
        name = data.get('name', '').strip()
        gain_dbi = float(data.get('gain_dbi', 0))
        
        if not key or not name:
            return jsonify({'error': 'Key and name required'}), 400
        
        antennas = load_antennas()
        antennas[key] = {
            'name': name,
            'gain_dbi': gain_dbi
        }
        save_antennas(antennas)
        
        return jsonify({'success': True, 'key': key})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/antennas/<key>', methods=['DELETE'])
@requires_auth
def delete_antenna(key):
    """Delete antenna"""
    try:
        antennas = load_antennas()
        key_lower = key.lower()
        
        if key_lower in antennas:
            del antennas[key_lower]
            save_antennas(antennas)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Antenna not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/custom-locations', methods=['GET'])
def get_custom_locations():
    """Get all custom saved locations"""
    cache = load_cache()
    custom_locs = {}
    
    for key, loc in cache.items():
        if loc.get('type') in ['custom', 'saved']:
            custom_locs[key] = loc
    
    return jsonify(custom_locs)

@bp.route('/api/custom-locations', methods=['POST'])
@requires_auth
def add_custom_location():
    """Add or update custom location"""
    try:
        data = request.json
        alias = data.get('alias', '').strip().lower()
        label = data.get('label', '').strip()
        lat = data.get('lat')
        lon = data.get('lon')
        grid = data.get('grid', '')
        
        if not alias:
            return jsonify({'error': 'Alias required'}), 400
        
        if lat is None or lon is None:
            return jsonify({'error': 'Coordinates required'}), 400
        
        cache = load_cache()
        cache[alias] = {
            'lat': float(lat),
            'lon': float(lon),
            'label': label or alias,
            'grid': grid,
            'type': 'custom',
            'saved_date': datetime.now().isoformat()
        }
        save_cache(cache)
        
        return jsonify({'success': True, 'alias': alias})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/custom-locations/<alias>', methods=['DELETE'])
@requires_auth
def delete_custom_location(alias):
    """Delete custom location"""
    try:
        cache = load_cache()
        alias_lower = alias.lower()
        
        if alias_lower in cache:
            del cache[alias_lower]
            save_cache(cache)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Location not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/snap-to-peak', methods=['POST'])
def snap_to_peak():
    """Find highest elevation within 500m radius of given coordinates"""
    if not SRTM_AVAILABLE:
        return jsonify({'error': 'SRTM data not available'}), 500
    
    try:
        data = request.json
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        radius_m = float(data.get('radius', 500))
        
        elevation_data = srtm.get_data()
        
        # Search grid around point (approximate: 1 degree ~ 111km)
        radius_deg = radius_m / 111000
        search_points = 20  # Resolution of search grid
        
        max_elevation = elevation_data.get_elevation(lat, lon) or 0
        best_lat = lat
        best_lon = lon
        
        # Search in a grid pattern
        for dlat in np.linspace(-radius_deg, radius_deg, search_points):
            for dlon in np.linspace(-radius_deg, radius_deg, search_points):
                test_lat = lat + dlat
                test_lon = lon + dlon
                
                # Check if within radius
                dist_m = haversine_distance(lat, lon, test_lat, test_lon) * 1000
                if dist_m <= radius_m:
                    elev = elevation_data.get_elevation(test_lat, test_lon)
                    if elev and elev > max_elevation:
                        max_elevation = elev
                        best_lat = test_lat
                        best_lon = test_lon
        
        return jsonify({
            'success': True,
            'original': {'lat': lat, 'lon': lon},
            'peak': {'lat': best_lat, 'lon': best_lon, 'elevation': max_elevation},
            'moved_distance': haversine_distance(lat, lon, best_lat, best_lon) * 1000
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/resolve-location', methods=['POST'])
def resolve_location():
    """Resolve a location (callsign/grid/node/coords) to coordinates"""
    try:
        data = request.json
        location = data.get('location', '').strip()
        
        if not location:
            return jsonify({'error': 'Location required'}), 400
        
        cache = load_cache()
        info = parse_location(location, cache)
        
        if not info:
            return jsonify({'error': f'Could not resolve location: {location}'}), 400
        
        # Save to cache if it was a successful lookup
        save_cache(cache)
        
        return jsonify({
            'success': True,
            'location': info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/settings', methods=['GET'])
def get_settings():
    """Get user settings"""
    settings = load_settings()
    return jsonify(settings)

@bp.route('/api/settings', methods=['POST'])
@requires_auth
def update_settings():
    """Update user settings"""
    try:
        settings = request.json
        save_settings(settings)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/nodes', methods=['POST'])
@requires_auth
def add_node():
    """Add custom node alias"""
    try:
        data = request.json
        alias = data.get('alias', '').lower().strip()
        name = data.get('name', '')
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        callsign = data.get('callsign', '')
        
        if not alias:
            return jsonify({'error': 'Alias required'}), 400
        
        settings = load_settings()
        settings['node_aliases'][alias] = {
            'callsign': callsign,
            'name': name,
            'lat': lat,
            'lon': lon
        }
        save_settings(settings)
        
        return jsonify({'success': True, 'alias': alias})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/nodes/<alias>', methods=['DELETE'])
@requires_auth
def delete_node(alias):
    """Delete custom node alias"""
    try:
        settings = load_settings()
        alias_lower = alias.lower()
        
        if alias_lower in settings['node_aliases']:
            del settings['node_aliases'][alias_lower]
            save_settings(settings)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Node not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add health endpoint
@bp.route("/health")
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "ok",
        "service": "vhf-tools",
        "version": "1.0.0",
        "srtm_available": SRTM_AVAILABLE,
        "data_dir": str(DATA_DIR),
        "timestamp": datetime.now().isoformat()
    }), 200


# Register blueprint
app.register_blueprint(bp)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    print("Starting VHF Path Analysis Web UI...")
    if URL_PREFIX:
        print(f"URL Prefix: {URL_PREFIX}")
        print(f"Access at: http://localhost:{port}{URL_PREFIX}/")
    else:
        print(f"Access at: http://localhost:{port}/")
    print(f"Data directory: {DATA_DIR}")
    print(f"SRTM available: {SRTM_AVAILABLE}")
    app.run(host="0.0.0.0", port=port, debug=True)
