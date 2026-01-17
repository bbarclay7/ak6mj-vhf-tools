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

**Address Geocoding:** Powered by OpenStreetMap (Nominatim) with Google Geocoding API fallback for street-level addresses. Enter any address, city, or landmark!

## Built-in Node Aliases

- `KBERR` - Berryessa Peak Digipeater (KE6YUV)
- `KJOHN` - Livermore Node (KF6ANX-4)
- `COOL` - Cool BBS Node (KM6LYW-4)

## Documentation

See `/Users/bb/work/srtm/VHF_PATH_ANALYSIS_README.md` for detailed usage guide.

## Configuration

### Google Geocoding API (Optional)

For street-level address geocoding, you can optionally set up a Google Geocoding API key:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the "Geocoding API"
4. Create credentials (API Key)
5. Set the environment variable:
   ```bash
   export GOOGLE_GEOCODING_API_KEY="your-api-key-here"
   ```

**Note:** Google offers $200/month free credit (~40,000 geocoding requests). Nominatim (OpenStreetMap) is always tried first and works for most cities/landmarks. Google is only used as a fallback for addresses not found by Nominatim.

## Deployment

Hosted at www.shoeph.one using:
- Flask with Blueprint architecture
- URL prefix support (`/vhf`)
- ProxyFix for reverse proxy
- Systemd service management
- Pre-cached SRTM data for CA/OR/WA
- Google Geocoding API for street-level addresses

## Credits

**Created by:** Brandon Brown, AK6MJ
**Data Sources:** NASA SRTM, FCC ULS Database
**Purpose:** Amateur radio VHF path planning and troubleshooting

## License

Free for amateur radio use. SRTM data is public domain (NASA/USGS).

**73 de AK6MJ** 📻
