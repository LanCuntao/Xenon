# XeOES - Online Windows EXE Build

This repo is prepared to build a Windows EXE via GitHub Actions.

## How to use
1) Create a new GitHub repo and upload all files in this folder.
2) Go to the **Actions** tab and run the workflow: **Build Windows EXE (XeOES)**.
3) When it finishes, download the artifact **XeOES-win**.
4) Run: `dist/XeOES/XeOES.exe`

## Notes
- This is a folder ("onedir") build for reliability with numpy/scipy/matplotlib.
- The backend script is bundled as data and loaded via resource_path in the GUI.
