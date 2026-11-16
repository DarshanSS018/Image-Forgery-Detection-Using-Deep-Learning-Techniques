"""
startup.py — Local development launcher for the Image Forgery Detection web app.

Usage:
    python startup.py

This script sets up the Python path so web_app/app.py can find sibling modules
(exif_utils, models_v2, batch_utils) regardless of where you run it from.
"""

import os
import sys

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_APP_DIR  = os.path.join(PROJECT_ROOT, 'web_app')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WEB_APP_DIR not in sys.path:
    sys.path.insert(0, WEB_APP_DIR)

# Change to web_app dir so Flask static/template resolution is correct
os.chdir(WEB_APP_DIR)

from app import app  # noqa: E402

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('\n' + '='*50)
    print('Image Forgery Detection — Web App')
    print(f'Open: http://localhost:{port}')
    print('='*50 + '\n')
    app.run(debug=True, host='0.0.0.0', port=port)
