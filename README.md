# AK6MJ VHF Tools

VHF path analysis and terrain profiling tools for amateur radio operators.

## Features

- 🗺️ **Terrain Analysis** - Real terrain data from NASA SRTM
- 📡 **Fresnel Zone Analysis** - See if terrain is blocking your RF path
- 📊 **Link Budget Calculator** - Calculate path loss and fade margins
- 🎯 **Callsign Lookup** - Automatic FCC database lookup
- 🏷️ **Node Aliases** - Built-in support for common packet nodes
- 💾 **Smart Caching** - Works offline after initial data fetch
- 🌐 **Web Interface** - User-friendly GUI, no CLI needed
- 📻 **Repeater Coverage Lookup** - Find reachable repeaters from your location with terrain-based analysis
- 💾 **CHIRP Export** - Download reachable repeaters as CHIRP-compatible CSV files

## Live Demo

**Hosted Version:** https://www.shoeph.one/vhf/

## Local Installation

```bash
# Clone repository
git clone https://github.com/bbarclay7/ak6mj-vhf-tools.git
cd ak6mj-vhf-tools

# Run with uv (recommended)
uv run vhf_webapp.py

# Or install dependencies and run
pip install -r requirements.txt
python vhf_webapp.py
```

Access at: http://localhost:5001

## Usage

Enter two locations (callsigns, grid squares, or coordinates) and the tool will:

1. Look up locations (from FCC database if callsign)
2. Calculate great circle distance and bearing
3. Compute link budget (path loss, fade margin)
4. Generate elevation profile with Fresnel zone analysis
5. Show if terrain is blocking your signal

### Example Inputs

- **Callsigns:** `AK6MJ` → `KM6LYW`
- **Addresses:** `Sacramento, CA` → `Mount Diablo, CA`
- **Cities/Landmarks:** `Berryessa Peak` → `Cool, CA`
- **Node Aliases:** `AK6MJ` → `KBERR` (Berryessa Peak digi)
- **Grid Squares:** `CM98jq` → `CM98kq`
- **Coordinates:** `38.6779,-121.1761` → `38.8894,-121.0156`

**Address Geocoding:** Powered by OpenStreetMap (Nominatim). Enter any address, city, or landmark!

## Repeater Coverage Lookup

The repeater coverage feature helps you discover which repeaters are reachable from your location using real terrain analysis:

### Setup

**RepeaterBook API Access** (free for non-commercial use):

The RepeaterBook API requires contact information for identification. You can provide this in two ways:

**Option 1: Email (Recommended for personal use)**
```bash
export REPEATERBOOK_EMAIL="your-email@example.com"
```

**Option 2: API Key (For registered API users)**
```bash
export REPEATERBOOK_API_KEY="your-api-key-here"
```

- Visit [RepeaterBook](https://www.repeaterbook.com/) to learn more about API access
- Register at [RepeaterBook Registration](https://www.repeaterbook.com/register) if needed
- See [API Documentation](https://www.repeaterbook.com/wiki/doku.php?id=api/) for details

### Usage

1. Enter your location (callsign, address, grid square, or coordinates)
2. Select your state (2-letter code, e.g., CA, WA)
3. Set search radius (default: 80km / ~50 miles)
4. Configure your radio parameters:
   - TX power (watts)
   - Antenna height (meters)
   - Antenna type
5. Select bands to search (2m, 70cm, etc.)
6. Click "Find Repeaters"

### Results

The tool will:
- Query RepeaterBook for nearby repeaters in the selected bands
- Calculate distance to each repeater
- Perform terrain analysis for line-of-sight determination
- Calculate link budget including:
  - Path loss
  - Terrain obstruction loss
  - Received signal strength
  - Fade margin
- Classify each repeater as:
  - ✓ **Clear** - Direct line-of-sight path
  - ⚠ **Marginal** - Partial Fresnel zone obstruction
  - ⚠ **Obstructed** - Significant terrain obstruction
  - ✗ **Blocked** - No viable path

### CHIRP Export

Export reachable repeaters to a CHIRP-compatible CSV file for easy radio programming:

1. Complete a repeater search
2. Click "Download CHIRP File"
3. Open the CSV file in CHIRP
4. Upload to your radio

The exported file includes:
- Frequency
- Offset (+ or -)
- CTCSS tone
- Callsign and location
- Distance from your location

## Built-in Node Aliases

- `KBERR` - Berryessa Peak Digipeater (KE6YUV)
- `KJOHN` - Livermore Node (KF6ANX-4)
- `COOL` - Cool BBS Node (KM6LYW-4)
- `WAQTH` - WA QTH - Freeland (AK6MJ)

## Documentation

See `/Users/bb/work/srtm/VHF_PATH_ANALYSIS_README.md` for detailed usage guide.

## Deployment

Hosted at www.shoeph.one using:
- Flask with Blueprint architecture
- URL prefix support (`/vhf`)
- ProxyFix for reverse proxy
- Systemd service management
- Pre-cached SRTM data for CA/OR/WA

## Credits

**Created by:** Brandon Brown, AK6MJ
**Data Sources:** NASA SRTM, FCC ULS Database
**Purpose:** Amateur radio VHF path planning and troubleshooting

## License

Free for amateur radio use. SRTM data is public domain (NASA/USGS).

**73 de AK6MJ** 📻
