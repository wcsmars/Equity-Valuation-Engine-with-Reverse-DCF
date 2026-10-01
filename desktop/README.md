# Equity Research Automation — macOS app

This Electron wrapper starts the Python backend and Next.js frontend on free
local ports, loads the dashboard in a native window, and stops both servers
when you quit. It uses the same optional API keys and data sources as the web
app.

## Run from source

Use Python 3.11+ and Node.js 22.12+ with npm. From the repository root (the
folder containing `run_dev.sh`):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r backend/requirements-backend.txt
npm --prefix frontend ci
npm --prefix desktop ci
npm --prefix desktop start
```

Copy `.env.example` to `.env` and add your own optional API keys. Set
`SEC_USER_AGENT` to your contact identity to enable SEC EDGAR requests. The
desktop app reads this file at each launch. It builds the frontend automatically
when a production build is missing, so the first launch can take longer.

Source launches find the project relative to `desktop/main.js`. No absolute
paths or generated configuration are needed. After moving the checkout,
recreate `.venv` because Python virtual environments are not relocatable.

`./run_dev.sh` is an alternative dependency-setup path: it installs dependencies
and starts development servers. Stop those servers with Ctrl-C before launching
the desktop app. It does not create a production frontend build.

Run `npm --prefix desktop test` from the repository root to check the runtime
helpers without opening a window. These tests use Node's standard library and
temporary local HTTP servers; they do not require Electron or API keys.

## Build a local app

After the setup above, run:

```bash
npm --prefix desktop run dist
```

The build generates an ignored `desktop/runtime-paths.json` with the current
checkout and Node executable paths, then packages the Electron wrapper. The
unsigned app appears under `desktop/dist/mac-arm64/` on Apple Silicon, or
`desktop/dist/mac/` on Intel. The `.app` itself can be moved to Applications.
For a disk image, run `npm --prefix desktop run dmg`.

The packaged app depends on this local checkout, its `.venv`, frontend
dependencies, and Node installation; it is not a self-contained distribution.
Rebuild after moving the checkout or changing the Node installation. Do not
publish the generated configuration or packaged local app as a portable release.
If macOS prompts about the unsigned app, use Finder's Open command and the
available macOS security controls for your own local build.

`ERC_PROJECT_ROOT` can override the checkout location for source runs or
packaging. `ERC_NODE_PATH` can select a Node executable. For example:

```bash
ERC_PROJECT_ROOT="/path/to/checkout" npm --prefix desktop run dist
```

When no configured Node executable is available, source runs use Electron's
Node runtime. Packaged runs use the Node path captured during the build, with
the same fallback if it is unavailable.

## Runtime behavior

The wrapper loads `.env` into the backend, starts `uvicorn backend.app:app`,
ensures `frontend/.next/BUILD_ID` exists, then starts `next start`. It waits for
both servers before showing the dashboard and passes the backend port to the
client through `?api=`, allowing direct requests to the local backend.

Running the web development server can replace the production build. The next
desktop launch detects the missing production build and rebuilds it. Exports
are saved in your Downloads folder and revealed in Finder. News and filing
links open in your default browser.

## Regenerate the icon

The existing icon assets are included. To change them, install Pillow in the
project environment and run these commands from `desktop/` on macOS:

```bash
../.venv/bin/python -m pip install Pillow
../.venv/bin/python make_icon.py
mkdir -p assets/icon.iconset
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" assets/icon_master.png --out "assets/icon.iconset/icon_${size}x${size}.png"
  double=$((size * 2))
  sips -z "$double" "$double" assets/icon_master.png --out "assets/icon.iconset/icon_${size}x${size}@2x.png"
done
iconutil -c icns assets/icon.iconset -o assets/icon.icns
```
