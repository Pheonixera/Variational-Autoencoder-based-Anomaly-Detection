"""
Flask Web Dashboard for Network Intrusion Detection System
Real-time monitoring interface with live updates
"""

from flask import Flask, render_template, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)

# FIXED: Use absolute path
DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'live_data.json')
print(f"🔍 Looking for data at: {DATA_FILE}")

def load_data():
    """Helper function to load data from file"""
    try:
        print(f"🔍 Checking if file exists: {os.path.exists(DATA_FILE)}")
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
            
            # Log what we loaded
            flows = len(data.get('recent_alerts', []))
            total = data.get('summary', {}).get('session_info', {}).get('total_flows', 0)
            print(f"✅ Loaded data: {flows} recent alerts, {total} total flows")
            
            return data
        else:
            print(f"❌ Data file NOT found at: {DATA_FILE}")
            # Return empty data structure
            return {
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'summary': {
                    'session_info': {
                        'total_flows': 0,
                        'total_alerts': 0,
                        'total_packets': 0
                    },
                    'statistics': {
                        'decisions': {},
                        'alert_levels': {},
                        'protocols': {}
                    }
                },
                'recent_alerts': [],
                'top_suspicious': []
            }
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """API endpoint for dashboard statistics"""
    print(f"\n📊 /api/stats called")
    data = load_data()
    if data:
        total_flows = data.get('summary', {}).get('session_info', {}).get('total_flows', 0)
        print(f"✅ Returning stats - Total flows: {total_flows}")
        return jsonify(data)
    else:
        print(f"❌ Failed to load data")
        return jsonify({'error': 'Failed to load data'}), 500

@app.route('/api/flows')
def get_flows():
    """API endpoint for all flow data"""
    print(f"\n📊 /api/flows called")
    data = load_data()
    if data:
        flows = data.get('recent_alerts', [])
        print(f"✅ Returning {len(flows)} flows")
        return jsonify(flows)
    else:
        print(f"❌ Failed to load data")
        return jsonify([])

@app.route('/api/alerts')
def get_alerts():
    """API endpoint for alerts"""
    print(f"\n📊 /api/alerts called")
    data = load_data()
    if data:
        alerts = data.get('recent_alerts', [])
        print(f"✅ Returning {len(alerts)} alerts")
        return jsonify(alerts)
    else:
        print(f"❌ Failed to load data")
        return jsonify([])  

if __name__ == '__main__':
    print("="*60)
    print("NIDS WEB DASHBOARD STARTING")
    print("="*60)
    print(f"\n✓ Dashboard URL: http://localhost:5000")
    print(f"✓ Data file path: {DATA_FILE}")
    print(f"✓ Data file exists: {os.path.exists(DATA_FILE)}")
    
    if os.path.exists(DATA_FILE):
        size = os.path.getsize(DATA_FILE)
        print(f"✓ Data file size: {size:,} bytes")
        
        # Load and preview data
        try:
            with open(DATA_FILE, 'r') as f:
                preview = json.load(f)
            flows = len(preview.get('recent_alerts', []))
            total = preview.get('summary', {}).get('session_info', {}).get('total_flows', 0)
            print(f"✓ Preview: {flows} recent alerts, {total} total flows")
        except Exception as e:
            print(f"✗ Could not preview data: {e}")
    else:
        print(f"✗ Data file NOT FOUND!")
        print(f"  Expected location: {DATA_FILE}")
        print(f"  Please run Cell 11 in notebook to create data file")
    
    print("\n✓ API Endpoints:")
    print("  - /              (Dashboard UI)")
    print("  - /api/stats     (Summary statistics)")
    print("  - /api/flows     (All flow data)")
    print("  - /api/alerts    (Alert data for table)")
    print("\n⚠️  Press Ctrl+C to stop server\n")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
