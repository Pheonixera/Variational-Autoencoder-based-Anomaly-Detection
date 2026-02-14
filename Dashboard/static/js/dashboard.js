// NIDS Dashboard JavaScript - Simplified & Debugged

console.log('🔵 JavaScript file loaded!');

let decisionChart, alertChart, protocolChart;

// Wait for page to load
window.addEventListener('load', function () {
    console.log('🟢 Page loaded, initializing dashboard...');
    initializeDashboard();
});

function initializeDashboard() {
    console.log('🟡 Initializing charts...');
    initializeCharts();

    console.log('🟡 Loading initial data...');
    loadDashboardData();

    // Auto-refresh every 5 seconds
    setInterval(function () {
        console.log('🔄 Auto-refreshing data...');
        loadDashboardData();
    }, 5000);
}

// Initialize Chart.js charts
function initializeCharts() {
    try {
        // Decision Results Chart
        const decisionCtx = document.getElementById('decisionChart');
        if (!decisionCtx) {
            console.error('❌ decisionChart canvas not found!');
            return;
        }

        decisionChart = new Chart(decisionCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Normal', 'Suspicious', 'Attack'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
        console.log('✅ Decision chart created');

        // Alert Levels Chart
        const alertCtx = document.getElementById('alertChart');
        alertChart = new Chart(alertCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Safe', 'Warning', 'Critical'],
                datasets: [{
                    label: 'Count',
                    data: [0, 0, 0],
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        console.log('✅ Alert chart created');

        // Protocol Distribution Chart
        const protocolCtx = document.getElementById('protocolChart');
        protocolChart = new Chart(protocolCtx.getContext('2d'), {
            type: 'pie',
            data: {
                labels: ['TCP', 'UDP', 'ICMP'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#667eea', '#764ba2', '#f59e0b']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
        console.log('✅ Protocol chart created');

    } catch (error) {
        console.error('❌ Error initializing charts:', error);
    }
}

// Load dashboard data from API
function loadDashboardData() {
    console.log('📡 Fetching data from /api/stats...');

    fetch('/api/stats')
        .then(response => {
            console.log('📥 Response status:', response.status);
            if (!response.ok) {
                throw new Error('HTTP error ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            console.log('📊 Data received:', data);

            if (data.error) {
                console.error('❌ API returned error:', data.error);
                updateStatus('Error', false);
                return;
            }

            updateStatistics(data);
            updateCharts(data);
            updateAlertsTable();
            updateStatus('Active', true);

            console.log('✅ Dashboard updated successfully');
        })
        .catch(error => {
            console.error('❌ Failed to load data:', error);
            updateStatus('Error', false);
        });
}

// Update statistics cards
function updateStatistics(data) {
    console.log('📈 Updating statistics...');

    try {
        const summary = data.summary || {};
        const sessionInfo = summary.session_info || {};
        const statistics = summary.statistics || {};

        const totalFlows = sessionInfo.total_flows || 0;
        const totalPackets = sessionInfo.total_packets || 0;
        const totalAlerts = sessionInfo.total_alerts || 0;
        const decisions = statistics.decisions || {};
        const normalFlows = decisions.NORMAL || 0;

        console.log('Values:', { totalFlows, totalPackets, totalAlerts, normalFlows });

        // Update DOM elements
        document.getElementById('totalFlows').textContent = totalFlows.toLocaleString();
        document.getElementById('totalPackets').textContent = totalPackets.toLocaleString();
        document.getElementById('totalAlerts').textContent = totalAlerts.toLocaleString();
        document.getElementById('normalFlows').textContent = normalFlows.toLocaleString();

        // Update last update time
        const now = new Date();
        document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();

        console.log('✅ Statistics updated');
    } catch (error) {
        console.error('❌ Error updating statistics:', error);
    }
}

// Update charts with new data
function updateCharts(data) {
    console.log('📊 Updating charts...');

    try {
        const statistics = data.summary?.statistics || {};

        // Update Decision Chart
        const decisions = statistics.decisions || {};
        const decisionData = [
            decisions.NORMAL || 0,
            decisions.SUSPICIOUS || 0,
            decisions.ATTACK || 0
        ];
        console.log('Decision data:', decisionData);

        decisionChart.data.datasets[0].data = decisionData;
        decisionChart.update();

        // Update Alert Levels Chart
        const alertLevels = statistics.alert_levels || {};
        const alertData = [
            alertLevels.SAFE || 0,
            alertLevels.WARNING || 0,
            alertLevels.CRITICAL || 0
        ];
        console.log('Alert data:', alertData);

        alertChart.data.datasets[0].data = alertData;
        alertChart.update();

        // Update Protocol Chart
        const protocols = statistics.protocols || {};
        const protocolData = [
            protocols.TCP || 0,
            protocols.UDP || 0,
            protocols.ICMP || 0
        ];
        console.log('Protocol data:', protocolData);

        protocolChart.data.datasets[0].data = protocolData;
        protocolChart.update();

        console.log('✅ Charts updated');
    } catch (error) {
        console.error('❌ Error updating charts:', error);
    }
}

// Update alerts table
function updateAlertsTable() {
    console.log('📋 Updating alerts table...');

    fetch('/api/alerts')
        .then(response => response.json())
        .then(alerts => {
            console.log('📥 Alerts received:', alerts.length);

            const tbody = document.getElementById('alertsTableBody');

            if (!alerts || alerts.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" class="loading">No alerts yet</td></tr>';
                console.log('⚠️ No alerts to display');
                return;
            }

            // Show last 20 alerts
            const recentAlerts = alerts.slice(-20).reverse();

            tbody.innerHTML = recentAlerts.map(alert => {
                return `
                    <tr>
                        <td>${formatTime(alert.timestamp)}</td>
                        <td><code>${truncateIP(alert.src_ip)}</code></td>
                        <td>${alert.dst_port}</td>
                        <td>${alert.protocol}</td>
                        <td>${alert.total_packets}</td>
                        <td>${alert.vae_error.toFixed(4)}</td>
                        <td>${alert.decision}</td>
                        <td><span class="badge badge-${alert.alert_level.toLowerCase()}">${alert.alert_level}</span></td>
                    </tr>
                `;
            }).join('');

            console.log('✅ Alerts table updated');
        })
        .catch(error => {
            console.error('❌ Failed to update alerts table:', error);
        });
}

// Update status indicator
function updateStatus(text, isActive) {
    document.getElementById('statusText').textContent = text;
    const dot = document.getElementById('statusDot');
    dot.style.background = isActive ? '#4ade80' : '#ef4444';
    console.log('Status updated:', text);
}

// Helper function to format timestamp
function formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    const parts = timestamp.split(' ');
    return parts.length > 1 ? parts[1] : timestamp;
}

// Helper function to truncate long IP addresses
function truncateIP(ip) {
    if (!ip) return 'N/A';
    if (ip.length > 30) {
        return ip.substring(0, 27) + '...';
    }
    return ip;
}

console.log('🔵 JavaScript file fully loaded!');
