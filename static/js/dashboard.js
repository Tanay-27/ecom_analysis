/**
 * Professional Analytics Dashboard JavaScript
 * Handles data fetching, visualization, and interactivity
 */

class AnalyticsDashboard {
    constructor() {
        this.charts = {};
        this.data = {};
        this.init();
    }

    async init() {
        try {
            await this.loadAllData();
            this.renderDashboard();
            this.hideLoadingSpinner();
            this.updateTimestamp();
            
            // Set up auto-refresh every 5 minutes
            setInterval(() => {
                this.refreshData();
            }, 300000);
            
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
            this.showError('Failed to load dashboard data. Please refresh the page.');
        }
    }

    async loadAllData() {
        const endpoints = [
            'business-overview',
            'sales-predictions',
            'ordering-recommendations',
            'inventory-status',
            'key-insights',
            'historical-analysis',
            'prediction-comparison',
            'comprehensive-ordering'
        ];

        const promises = endpoints.map(endpoint => 
            fetch(`/api/${endpoint}`)
                .then(response => response.json())
                .catch(error => {
                    console.error(`Failed to load ${endpoint}:`, error);
                    return { error: `Failed to load ${endpoint}` };
                })
        );

        const results = await Promise.all(promises);
        
        this.data = {
            businessOverview: results[0],
            salesPredictions: results[1],
            orderingRecommendations: results[2],
            inventoryStatus: results[3],
            keyInsights: results[4],
            historicalAnalysis: results[5],
            predictionComparison: results[6],
            comprehensiveOrdering: results[7]
        };
        
        // Load data freshness separately
        await this.loadDataFreshness();
    }

    async loadDataFreshness() {
        try {
            const response = await fetch('/api/data-freshness');
            const data = await response.json();
            
            if (data.last_updated) {
                const lastUpdated = new Date(data.last_updated);
                const now = new Date();
                const hoursAgo = Math.floor((now - lastUpdated) / (1000 * 60 * 60));
                
                let freshnessText;
                if (hoursAgo < 1) {
                    freshnessText = 'Updated just now';
                } else if (hoursAgo < 24) {
                    freshnessText = `Updated ${hoursAgo} hours ago`;
                } else {
                    const daysAgo = Math.floor(hoursAgo / 24);
                    freshnessText = `Updated ${daysAgo} days ago`;
                }
                
                const lastUpdatedElement = document.getElementById('lastUpdated');
                if (lastUpdatedElement) {
                    lastUpdatedElement.textContent = freshnessText;
                    lastUpdatedElement.title = `Data period: ${data.data_sources.sales_data?.period_covered || 'Unknown'}\nUpdate frequency: ${data.update_frequency}\nRecommended: ${data.recommended_update_frequency}`;
                }
            }
        } catch (error) {
            console.error('Error loading data freshness:', error);
            const lastUpdatedElement = document.getElementById('lastUpdated');
            if (lastUpdatedElement) {
                lastUpdatedElement.textContent = 'Data freshness unavailable';
            }
        }
    }

    renderDashboard() {
        this.renderMetrics();
        this.renderCharts();
        this.renderAlerts();
        this.renderTables();
        this.setupTabs();
        this.setupModals();
    }

    renderMetrics() {
        const data = this.data.businessOverview;
        if (data.error) return;

        // Update metric values
        document.getElementById('totalRevenue').textContent = this.formatCurrency(data.total_revenue);
        document.getElementById('totalOrders').textContent = this.formatNumber(data.total_orders);
        document.getElementById('activeSKUs').textContent = this.formatNumber(data.unique_skus);
        document.getElementById('avgOrderValue').textContent = `Avg: ${this.formatCurrency(data.avg_order_value)}`;
        
        // Update growth rate
        const growthElement = document.getElementById('revenueChange');
        const growthRate = data.growth_rate || 0;
        growthElement.textContent = `${growthRate >= 0 ? '+' : ''}${growthRate.toFixed(1)}%`;
        growthElement.className = `metric-change ${growthRate < 0 ? 'negative' : ''}`;

        // Update predictions
        const predictionsData = this.data.salesPredictions;
        if (!predictionsData.error) {
            document.getElementById('totalPredicted').textContent = this.formatNumber(predictionsData.total_predicted);
        }
    }

    renderCharts() {
        this.renderRevenueChart();
        this.renderTopSKUsChart();
        this.renderConfidenceChart();
        this.renderInventoryChart();
        this.renderPredictionComparisonChart();
        this.renderHistoricalRevenueChart();
        this.renderSeasonalChart();
        this.renderOrderUrgencyChart();
    }

    renderRevenueChart() {
        const data = this.data.businessOverview;
        if (data.error || !data.monthly_trend) return;

        const ctx = document.getElementById('revenueChart').getContext('2d');
        
        if (this.charts.revenue) {
            this.charts.revenue.destroy();
        }

        this.charts.revenue = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.monthly_trend.map(item => this.formatMonth(item.month)),
                datasets: [{
                    label: 'Revenue (₹)',
                    data: data.monthly_trend.map(item => item.revenue),
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#2563eb',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#2563eb',
                        borderWidth: 1,
                        callbacks: {
                            label: (context) => `Revenue: ${this.formatCurrency(context.raw)}`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            callback: (value) => this.formatCurrencyShort(value)
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    renderTopSKUsChart() {
        const data = this.data.businessOverview;
        if (data.error || !data.top_skus) return;

        const ctx = document.getElementById('topSKUsChart').getContext('2d');
        
        if (this.charts.topSKUs) {
            this.charts.topSKUs.destroy();
        }

        const colors = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

        this.charts.topSKUs = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.top_skus.map(item => `${item.sku} (${item.product_name || 'Unknown'})`),
                datasets: [{
                    data: data.top_skus.map(item => item.revenue),
                    backgroundColor: colors,
                    borderWidth: 0,
                    hoverBorderWidth: 2,
                    hoverBorderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        callbacks: {
                            label: (context) => {
                                const label = context.label;
                                const value = this.formatCurrency(context.raw);
                                const percentage = ((context.raw / data.top_skus.reduce((sum, item) => sum + item.revenue, 0)) * 100).toFixed(1);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    renderConfidenceChart() {
        const data = this.data.salesPredictions;
        if (data.error || !data.confidence_distribution) return;

        const ctx = document.getElementById('confidenceChart').getContext('2d');
        
        if (this.charts.confidence) {
            this.charts.confidence.destroy();
        }

        const confidenceData = data.confidence_distribution;
        const labels = Object.keys(confidenceData);
        const values = Object.values(confidenceData);
        const colors = {
            'High': '#10b981',
            'Medium': '#f59e0b',
            'Low': '#ef4444'
        };

        this.charts.confidence = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Number of SKUs',
                    data: values,
                    backgroundColor: labels.map(label => colors[label] || '#64748b'),
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            stepSize: 1
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    renderInventoryChart() {
        const data = this.data.inventoryStatus;
        console.log('Inventory Status Data:', data);
        if (data.error || !data.summary) {
            console.log('No inventory summary data available:', data.error || 'summary missing');
            return;
        }

        const ctx = document.getElementById('inventoryChart').getContext('2d');
        
        if (this.charts.inventory) {
            this.charts.inventory.destroy();
        }

        const summary = data.summary;
        const labels = ['Critical Items', 'High Priority', 'Medium Priority', 'Low Priority'];
        const values = [
            summary.critical_items || 0,
            summary.high_priority_items || 0, 
            summary.medium_priority_items || 0,
            summary.low_priority_items || 0
        ];
        const colors = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981'];

        this.charts.inventory = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
                    borderWidth: 0,
                    hoverBorderWidth: 2,
                    hoverBorderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        callbacks: {
                            label: (context) => {
                                const label = context.label;
                                const value = context.raw;
                                const total = values.reduce((sum, val) => sum + val, 0);
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return `${label}: ${value} SKUs (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    renderAlerts() {
        const data = this.data.keyInsights;
        if (data.error || !data.insights) return;

        const container = document.getElementById('alertsContainer');
        container.innerHTML = '';

        data.insights.forEach(insight => {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert ${insight.type} fade-in`;
            
            const iconMap = {
                critical: 'fas fa-exclamation-triangle',
                warning: 'fas fa-exclamation-circle',
                info: 'fas fa-info-circle',
                positive: 'fas fa-check-circle'
            };

            alertDiv.innerHTML = `
                <i class="${iconMap[insight.type] || 'fas fa-info-circle'}"></i>
                <span>${insight.message}</span>
            `;
            
            container.appendChild(alertDiv);
        });

        if (data.insights.length === 0) {
            container.innerHTML = `
                <div class="alert positive fade-in">
                    <i class="fas fa-check-circle"></i>
                    <span>All systems operating normally - no critical issues detected.</span>
                </div>
            `;
        }
    }

    renderTables() {
        this.renderCriticalOrdersTable();
        this.renderPredictionsTable();
        this.renderCriticalInventoryTable();
        this.renderHighPriorityTable();
    }

    renderCriticalOrdersTable() {
        const data = this.data.orderingRecommendations;
        if (data.error || !data.critical_orders) return;

        const tbody = document.querySelector('#criticalOrdersTable tbody');
        tbody.innerHTML = '';
        
        data.critical_orders.slice(0, 5).forEach(order => {
            const row = document.createElement('tr');
            row.className = 'slide-in';
            row.innerHTML = `
                <td>
                    <div class="sku-info">
                        <strong>${order.sku}</strong>
                        <small class="product-name">${order.product_name || 'Unknown Product'}</small>
                    </div>
                </td>
                <td><span class="status-badge critical">${order.urgency}</span></td>
                <td>${order.recommended_qty}</td>
                <td>${order.days_remaining} days</td>
            `;
            tbody.appendChild(row);
        });
    }

    renderPredictionsTable() {
        const data = this.data.salesPredictions;
        console.log('Predictions Table Data:', data);
        if (data.error || !data.predictions) {
            console.log('No predictions data for table:', data.error || 'predictions missing');
            return;
        }

        const tbody = document.querySelector('#predictionsTable tbody');
        const countElement = document.getElementById('predictionsCount');
        
        tbody.innerHTML = '';
        countElement.textContent = `${data.predictions.length} of 91 SKUs`;
        console.log('Rendering predictions table with', data.predictions.length, 'items');

        data.predictions.forEach((prediction, index) => {
            console.log('Processing prediction', index, ':', prediction);
            const row = document.createElement('tr');
            row.className = 'slide-in';
            row.innerHTML = `
                <td>
                    <div class="sku-info">
                        <strong>${prediction.sku}</strong>
                        <small class="product-name">${prediction.product_name || 'Unknown Product'}</small>
                    </div>
                </td>
                <td>${this.formatNumber(prediction.monthly_quantity || 0)}</td>
                <td>${(prediction.daily_average || 0).toFixed(1)}</td>
                <td>${((prediction.growth_rate || 0) * 100).toFixed(1)}%</td>
                <td><span class="status-badge ${(prediction.confidence || 'unknown').toLowerCase()}">${prediction.confidence || 'Unknown'}</span></td>
            `;
            tbody.appendChild(row);
        });
    }

    renderCriticalInventoryTable() {
        const data = this.data.inventoryStatus;
        if (data.error || !data.critical_items) return;

        const tbody = document.querySelector('#criticalInventoryTable tbody');
        const countElement = document.getElementById('criticalInventoryCount');
        
        tbody.innerHTML = '';
        countElement.textContent = `${data.critical_items.length} items`;

        data.critical_items.forEach(item => {
            const row = document.createElement('tr');
            row.className = 'slide-in';
            row.innerHTML = `
                <td>
                    <div class="sku-info">
                        <strong>${item.sku}</strong>
                        <small class="product-name">${item.product_name || 'Unknown Product'}</small>
                    </div>
                </td>
                <td>${this.formatNumber(item.current_stock)}</td>
                <td><span class="status-badge critical">${item.days_remaining.toFixed(1)} days</span></td>
                <td>${item.daily_demand.toFixed(1)}</td>
                <td>${this.formatNumber(item.reorder_point)}</td>
            `;
            tbody.appendChild(row);
        });
    }

    renderHighPriorityTable() {
        const data = this.data.orderingRecommendations;
        if (data.error || !data.high_priority_orders) return;

        const tbody = document.querySelector('#highPriorityTable tbody');
        const countElement = document.getElementById('highPriorityCount');
        
        tbody.innerHTML = '';
        countElement.textContent = `${data.high_priority_orders.length} items`;

        data.high_priority_orders.forEach(order => {
            const row = document.createElement('tr');
            row.className = 'slide-in';
            row.innerHTML = `
                <td>
                    <div class="sku-info">
                        <strong>${order.sku}</strong>
                        <small class="product-name">${order.product_name || 'Unknown Product'}</small>
                    </div>
                </td>
                <td>${this.formatNumber(order.recommended_qty)}</td>
                <td>${this.formatCurrency(order.estimated_cost)}</td>
                <td><span class="status-badge medium">${order.days_remaining.toFixed(1)} days</span></td>
                <td>${this.formatNumber(order.current_stock)}</td>
            `;
            tbody.appendChild(row);
        });
    }

    async refreshData() {
        try {
            await this.loadAllData();
            this.renderDashboard();
            this.updateTimestamp();
            console.log('Dashboard data refreshed');
        } catch (error) {
            console.error('Failed to refresh data:', error);
        }
    }

    hideLoadingSpinner() {
        document.getElementById('loadingSpinner').style.display = 'none';
        document.getElementById('mainContent').style.display = 'block';
    }

    showError(message) {
        document.getElementById('loadingSpinner').innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle" style="color: #ef4444; font-size: 2rem; margin-bottom: 1rem;"></i>
                <p style="color: #ef4444; font-weight: 600;">${message}</p>
            </div>
        `;
    }

    updateTimestamp() {
        const now = new Date();
        const timestamp = now.toLocaleString();
        document.getElementById('lastUpdated').textContent = timestamp;
        document.getElementById('footerTimestamp').textContent = timestamp;
    }

    renderPredictionComparisonChart() {
        const data = this.data.predictionComparison;
        console.log('Prediction Comparison Data:', data);
        if (data.error || !data.comparisons) {
            console.log('No prediction comparison data available:', data.error || 'comparisons missing');
            return;
        }

        const ctx = document.getElementById('predictionComparisonChart');
        if (!ctx) return;
        
        if (this.charts.predictionComparison) {
            this.charts.predictionComparison.destroy();
        }

        const comparisonData = data.comparisons.slice(0, 10); // Top 10 SKUs
        
        this.charts.predictionComparison = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: comparisonData.map(item => `${item.sku}`),
                datasets: [{
                    label: 'Actual Quantity',
                    data: comparisonData.map(item => item.actual_quantity),
                    backgroundColor: 'rgba(37, 99, 235, 0.7)',
                    borderColor: '#2563eb',
                    borderWidth: 1
                }, {
                    label: 'Predicted Quantity',
                    data: comparisonData.map(item => item.predicted_quantity),
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderColor: '#10b981',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    renderHistoricalRevenueChart() {
        const data = this.data.historicalAnalysis;
        console.log('Historical Analysis Data:', data);
        if (data.error || !data.yearly_trend) {
            console.log('No yearly trend data available:', data.error || 'yearly_trend missing');
            return;
        }

        const ctx = document.getElementById('historicalRevenueChart');
        if (!ctx) return;
        
        if (this.charts.historicalRevenue) {
            this.charts.historicalRevenue.destroy();
        }

        this.charts.historicalRevenue = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: data.yearly_trend.map(item => item.year.toString()),
                datasets: [{
                    label: 'Annual Revenue (₹)',
                    data: data.yearly_trend.map(item => item.revenue),
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#8b5cf6',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        callbacks: {
                            label: (context) => `Revenue: ${this.formatCurrency(context.raw)}`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            callback: (value) => this.formatCurrencyShort(value)
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    renderSeasonalChart() {
        const data = this.data.historicalAnalysis;
        console.log('Seasonal Analysis Data:', data);
        if (data.error || !data.seasonal_patterns) {
            console.log('No seasonal patterns data available:', data.error || 'seasonal_patterns missing');
            return;
        }

        const ctx = document.getElementById('seasonalChart');
        if (!ctx) return;
        
        if (this.charts.seasonal) {
            this.charts.seasonal.destroy();
        }

        this.charts.seasonal = new Chart(ctx.getContext('2d'), {
            type: 'radar',
            data: {
                labels: data.seasonal_patterns.map(item => item.month),
                datasets: [{
                    label: 'Average Revenue',
                    data: data.seasonal_patterns.map(item => item.avg_revenue),
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: '#f59e0b',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        callbacks: {
                            label: (context) => `Avg Revenue: ${this.formatCurrency(context.raw)}`
                        }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    renderOrderUrgencyChart() {
        const data = this.data.inventoryStatus;
        console.log('Order Urgency Data:', data);
        if (data.error || !data.urgency_distribution) {
            console.log('No urgency distribution data available:', data.error || 'urgency_distribution missing');
            return;
        }

        const ctx = document.getElementById('orderUrgencyChart');
        if (!ctx) return;
        
        if (this.charts.orderUrgency) {
            this.charts.orderUrgency.destroy();
        }

        const urgencyDist = data.urgency_distribution;
        const urgencyData = [
            urgencyDist.critical || 0,
            urgencyDist.high || 0,
            urgencyDist.medium || 0,
            urgencyDist.low || 0
        ];

        this.charts.orderUrgency = new Chart(ctx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Critical (Order Today)', 'High Priority', 'Medium Priority', 'Low Priority'],
                datasets: [{
                    data: urgencyData,
                    backgroundColor: ['#ef4444', '#f59e0b', '#3b82f6', '#10b981'],
                    borderWidth: 0,
                    hoverBorderWidth: 2,
                    hoverBorderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        callbacks: {
                            label: (context) => {
                                const label = context.label;
                                const value = context.raw;
                                const total = urgencyData.reduce((sum, val) => sum + val, 0);
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return `${label}: ${value} SKUs (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-tab');
                
                // Remove active class from all buttons and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked button and corresponding content
                button.classList.add('active');
                document.getElementById(targetTab).classList.add('active');
            });
        });
    }

    setupModals() {
        // Setup clickable cards
        const clickableCards = document.querySelectorAll('.metric-card.clickable');
        clickableCards.forEach(card => {
            card.addEventListener('click', () => {
                const modalId = card.getAttribute('data-modal');
                this.openModal(modalId);
            });
        });

        // Setup modal close buttons
        const closeButtons = document.querySelectorAll('.close');
        closeButtons.forEach(button => {
            button.addEventListener('click', () => {
                this.closeAllModals();
            });
        });

        // Close modal when clicking outside
        window.addEventListener('click', (event) => {
            if (event.target.classList.contains('modal')) {
                this.closeAllModals();
            }
        });
    }

    openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'block';
            this.populateModal(modalId);
        }
    }

    closeAllModals() {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            modal.style.display = 'none';
        });
    }

    populateModal(modalId) {
        switch(modalId) {
            case 'revenue-modal':
                this.populateRevenueModal();
                break;
            case 'orders-modal':
                this.populateOrdersModal();
                break;
            case 'skus-modal':
                this.populateSKUsModal();
                break;
            case 'predictions-modal':
                this.populatePredictionsModal();
                break;
        }
    }

    populateRevenueModal() {
        const data = this.data.businessOverview;
        if (data.error || !data.monthly_trend) return;

        const tbody = document.getElementById('revenueModalTable');
        tbody.innerHTML = '';

        data.monthly_trend.forEach((item, index) => {
            const prevRevenue = index > 0 ? data.monthly_trend[index - 1].revenue : item.revenue;
            const growth = ((item.revenue - prevRevenue) / prevRevenue * 100).toFixed(1);
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.month}</td>
                <td>${this.formatCurrency(item.revenue)}</td>
                <td>${this.formatNumber(item.quantity)}</td>
                <td><span class="status-badge ${growth >= 0 ? 'high' : 'low'}">${growth >= 0 ? '+' : ''}${growth}%</span></td>
            `;
            tbody.appendChild(row);
        });
    }

    populateOrdersModal() {
        const data = this.data.businessOverview;
        if (data.error || !data.top_skus) return;

        const tbody = document.getElementById('ordersModalTable');
        tbody.innerHTML = '';

        data.top_skus.forEach(sku => {
            const avgOrderValue = sku.revenue / (sku.orders || 1);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${sku.sku}</strong></td>
                <td>${this.formatNumber(sku.orders || 'N/A')}</td>
                <td>${this.formatCurrency(sku.revenue)}</td>
                <td>${this.formatCurrency(avgOrderValue)}</td>
            `;
            tbody.appendChild(row);
        });
    }

    populateSKUsModal() {
        const data = this.data.businessOverview;
        if (data.error || !data.top_skus) return;

        const tbody = document.getElementById('skusModalTable');
        tbody.innerHTML = '';

        data.top_skus.forEach((sku, index) => {
            const performance = index < 2 ? 'Excellent' : index < 4 ? 'Good' : 'Average';
            const performanceClass = index < 2 ? 'high' : index < 4 ? 'medium' : 'low';
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>
                    <div class="sku-info">
                        <strong>${sku.sku}</strong>
                        <small class="product-name">${sku.product_name || 'Unknown Product'}</small>
                    </div>
                </td>
                <td>${this.formatCurrency(sku.revenue)}</td>
                <td>${this.formatNumber(sku.quantity || 'N/A')}</td>
                <td>${this.formatCurrency(sku.avg_price || 0)}</td>
                <td><span class="status-badge ${performanceClass}">${performance}</span></td>
            `;
            tbody.appendChild(row);
        });
    }

    populatePredictionsModal() {
        // Use sales predictions data instead of prediction comparison
        const data = this.data.salesPredictions;
        console.log('Predictions Modal Data:', data);
        if (data.error || !data.predictions) {
            console.log('No sales predictions data for modal:', data.error || 'predictions missing');
            return;
        }

        const tbody = document.getElementById('predictionsModalTable');
        tbody.innerHTML = '';

        data.predictions.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${item.sku}</strong></td>
                <td>${this.formatNumber(item.monthly_quantity / 30)}</td>
                <td>${this.formatNumber(item.monthly_quantity)}</td>
                <td><span class="status-badge ${item.confidence.toLowerCase()}">${item.confidence}</span></td>
                <td>${((item.growth_rate || 0) * 100).toFixed(1)}%</td>
            `;
            tbody.appendChild(row);
        });
    }

    renderTables() {
        this.renderCriticalOrdersTable();
        this.renderPredictionsTable();
        this.renderCriticalInventoryTable();
        this.renderHighPriorityTable();
        this.renderAllPredictionsTable();
        this.renderComprehensiveOrderingTable();
    }

    renderAllPredictionsTable() {
        const data = this.data.predictionComparison;
        console.log('All Predictions Table Data:', data);
        if (data.error || !data.comparisons) {
            console.log('No prediction comparison data for table:', data.error || 'comparisons missing');
            return;
        }

        const tbody = document.querySelector('#allPredictionsTable tbody');
        const countElement = document.getElementById('allPredictionsCount');
        
        tbody.innerHTML = '';
        countElement.textContent = `${data.comparisons.length} items`;

        data.comparisons.forEach(item => {
            const row = document.createElement('tr');
            row.className = 'slide-in';
            row.innerHTML = `
                <td><strong>${item.sku}</strong></td>
                <td>${this.formatNumber(item.actual_quantity)}</td>
                <td>${this.formatNumber(item.predicted_quantity.toFixed(0))}</td>
                <td>${item.predicted_quantity > 0 ? (item.predicted_quantity / 30).toFixed(1) : '0'}</td>
                <td>${item.accuracy_percent}%</td>
                <td><span class="status-badge ${item.confidence.toLowerCase()}">${item.confidence}</span></td>
                <td>${item.variance_percent > 0 ? '+' : ''}${item.variance_percent.toFixed(1)}%</td>
            `;
            tbody.appendChild(row);
        });
    }

    renderComprehensiveOrderingTable() {
        const data = this.data.comprehensiveOrdering;
        if (data.error || !data.comprehensive_data) return;

        const tbody = document.querySelector('#comprehensiveOrderingTable tbody');
        const countElement = document.getElementById('comprehensiveOrderingCount');
        if (!tbody || !countElement) return;

        tbody.innerHTML = '';
        countElement.textContent = `${data.comprehensive_data.length} items`;

        data.comprehensive_data.forEach(item => {
            const godownDist = (item.godown_distribution || [])
                .map(g => `${g.godown}(${g.quantity})`).join(', ');
            const row = document.createElement('tr');
            row.className = 'slide-in';
            row.innerHTML = `
                <td>
                    <div class="sku-info">
                        <strong>${item.sku}</strong>
                        <small class="product-name">${item.product_name || 'Unknown Product'}</small>
                    </div>
                </td>
                <td>${this.formatNumber(item.current_monthly_quantity)}</td>
                <td>${this.formatNumber(item.predicted_monthly_quantity)}</td>
                <td><span class="growth-indicator ${item.growth_rate_percent >= 0 ? 'positive' : 'negative'}">${item.growth_rate_percent}%</span></td>
                <td>${item.lead_time_days} days</td>
                <td><strong>${this.formatNumber(item.recommended_order_quantity)}</strong></td>
                <td>${this.formatNumber(item.current_stock)}</td>
                <td><span class="days-remaining ${item.days_remaining < 7 ? 'critical' : item.days_remaining < 14 ? 'warning' : 'good'}">${item.days_remaining} days</span></td>
                <td><span class="status-badge ${item.urgency ? item.urgency.toLowerCase() : 'medium'}">${item.urgency || 'Medium'}</span></td>
                <td>${this.formatCurrency(item.estimated_cost_inr)}</td>
                <td><span class="status-badge ${item.confidence ? item.confidence.toLowerCase() : 'medium'}">${item.confidence || 'Medium'}</span></td>
                <td class="godown-distribution">${godownDist || 'No data'}</td>
            `;
            tbody.appendChild(row);
        });
    }

    // Utility functions
    formatCurrency(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return '₹0';
        }

        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(value);
    }

    formatCurrencyShort(value) {
        if (value >= 10000000) {
            return `₹${(value / 10000000).toFixed(1)}Cr`;
        } else if (value >= 100000) {
            return `₹${(value / 100000).toFixed(1)}L`;
        } else if (value >= 1000) {
            return `₹${(value / 1000).toFixed(1)}K`;
        }

        return `₹${Math.round(value || 0)}`;
    }

    formatNumber(value) {
        if (value === null || value === undefined || value === 'N/A') {
            return '0';
        }

        const numericValue = Number(value);
        if (Number.isNaN(numericValue)) {
            return value;
        }

        return new Intl.NumberFormat('en-IN').format(numericValue);
    }

    formatMonth(monthString) {
        if (!monthString || !monthString.includes('-')) {
            return monthString || '';
        }

        const [year, month] = monthString.split('-');
        const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const monthIndex = parseInt(month, 10) - 1;

        if (Number.isNaN(monthIndex) || monthIndex < 0 || monthIndex > 11) {
            return monthString;
        }

        return `${monthNames[monthIndex]} ${year}`;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AnalyticsDashboard();
});

// Global function for CSV export
function exportComprehensiveCSV() {
    window.open('/export/comprehensive-ordering-csv', '_blank');
}

function exportOrderingScheduleCSV() {
    window.open('/export/ordering-schedule-csv', '_blank');
}

function exportOrderingScheduleExcel() {
    window.open('/export/ordering-schedule-excel', '_blank');
}

// Add keyboard shortcuts and click outside handler
document.addEventListener('keydown', (event) => {
    if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
            case 'r':
                event.preventDefault();
                location.reload();
                break;
            case 'p':
                event.preventDefault();
                window.print();
                break;
        }
    }
});

// Close export dropdowns when clicking outside
document.addEventListener('click', (event) => {
    if (!event.target.closest('.export-dropdown')) {
        document.querySelectorAll('.export-menu').forEach(menu => {
            menu.style.display = 'none';
        });
    }
});
