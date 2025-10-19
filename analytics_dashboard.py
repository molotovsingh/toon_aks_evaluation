"""
Analytics Dashboard for Legal Events Pipeline

Visualizes pipeline execution metrics from DuckDB database:
- Performance analysis (speed, quality)
- Cost tracking and efficiency
- Run history explorer with filters
- Summary statistics and trends

Usage: uv run streamlit run analytics_dashboard.py
"""

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Page config
st.set_page_config(
    page_title="Pipeline Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DB_PATH = Path(__file__).parent / "runs.duckdb"
CACHE_TTL = 60  # seconds


# ============================================================================
# Database Connection
# ============================================================================

@st.cache_resource
def get_db_connection():
    """Get cached read-only DuckDB connection"""
    if not DB_PATH.exists():
        st.error(f"❌ Database not found at `{DB_PATH}`")
        st.info("""
        **Setup Instructions**:
        1. Ensure you've run the ingestion script:
           ```bash
           uv run python scripts/ingest_metadata_to_duckdb.py \\
             --db runs.duckdb \\
             --glob "output/**/*_metadata.json"
           ```
        2. Refresh this page
        """)
        st.stop()

    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        return conn
    except Exception as e:
        st.error(f"❌ Failed to connect to database: {e}")
        st.stop()


# ============================================================================
# Query Functions (Cached)
# ============================================================================

@st.cache_data(ttl=CACHE_TTL)
def get_summary_stats() -> dict:
    """Get overall summary statistics"""
    conn = get_db_connection()

    query = """
    SELECT
        COUNT(*) as total_runs,
        COUNT(DISTINCT provider_name) as total_providers,
        COUNT(DISTINCT provider_model) as total_models,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
        ROUND(COALESCE(SUM(cost_usd), 0), 4) as total_cost,
        ROUND(AVG(events_extracted), 1) as avg_events,
        ROUND(AVG(extractor_seconds), 2) as avg_extraction_time,
        MIN(timestamp) as first_run,
        MAX(timestamp) as last_run
    FROM pipeline_runs
    """

    result = conn.execute(query).fetchone()

    return {
        'total_runs': result[0],
        'total_providers': result[1],
        'total_models': result[2],
        'success_count': result[3],
        'success_rate': round((result[3] / result[0] * 100) if result[0] > 0 else 0, 1),
        'total_cost': result[4],
        'avg_events': result[5],
        'avg_extraction_time': result[6],
        'first_run': result[7],
        'last_run': result[8]
    }


@st.cache_data(ttl=CACHE_TTL)
def get_recent_runs(limit: int = 10) -> pd.DataFrame:
    """Get most recent pipeline runs"""
    conn = get_db_connection()

    query = f"""
    SELECT
        run_id,
        timestamp,
        provider_name,
        provider_model,
        input_filename,
        events_extracted,
        ROUND(total_seconds, 2) as total_seconds,
        ROUND(cost_usd, 4) as cost_usd,
        status
    FROM pipeline_runs
    ORDER BY timestamp DESC
    LIMIT {limit}
    """

    return conn.execute(query).df()


@st.cache_data(ttl=CACHE_TTL)
def get_performance_by_model() -> pd.DataFrame:
    """Get performance metrics grouped by model"""
    conn = get_db_connection()

    query = """
    SELECT
        provider_model,
        provider_name,
        COUNT(*) as run_count,
        ROUND(AVG(extractor_seconds), 2) as avg_extraction_time,
        ROUND(MIN(extractor_seconds), 2) as min_extraction_time,
        ROUND(MAX(extractor_seconds), 2) as max_extraction_time,
        ROUND(AVG(events_extracted), 1) as avg_events,
        ROUND(AVG(citations_found), 1) as avg_citations
    FROM pipeline_runs
    WHERE status = 'success'
      AND extractor_seconds IS NOT NULL
      AND events_extracted IS NOT NULL
    GROUP BY provider_model, provider_name
    ORDER BY avg_extraction_time ASC
    """

    return conn.execute(query).df()


@st.cache_data(ttl=CACHE_TTL)
def get_cost_breakdown() -> pd.DataFrame:
    """Get cost breakdown by provider and model"""
    conn = get_db_connection()

    query = """
    SELECT
        provider_name,
        provider_model,
        COUNT(*) as run_count,
        ROUND(SUM(cost_usd), 4) as total_cost,
        ROUND(AVG(cost_usd), 4) as avg_cost_per_run,
        ROUND(SUM(cost_usd) / NULLIF(SUM(events_extracted), 0), 4) as cost_per_event,
        SUM(events_extracted) as total_events
    FROM pipeline_runs
    WHERE cost_usd IS NOT NULL
      AND cost_usd > 0
    GROUP BY provider_name, provider_model
    ORDER BY total_cost DESC
    """

    return conn.execute(query).df()


@st.cache_data(ttl=CACHE_TTL)
def get_time_series_data(days: int = 30) -> pd.DataFrame:
    """Get daily aggregated metrics for trend charts"""
    conn = get_db_connection()

    query = f"""
    SELECT
        DATE_TRUNC('day', timestamp) as date,
        COUNT(*) as runs,
        ROUND(AVG(extractor_seconds), 2) as avg_extraction_time,
        ROUND(SUM(cost_usd), 4) as daily_cost,
        SUM(events_extracted) as total_events
    FROM pipeline_runs
    WHERE timestamp >= CURRENT_DATE - INTERVAL '{days} days'
    GROUP BY DATE_TRUNC('day', timestamp)
    ORDER BY date ASC
    """

    return conn.execute(query).df()


@st.cache_data(ttl=CACHE_TTL)
def get_filtered_runs(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None
) -> pd.DataFrame:
    """Get filtered runs for explorer"""
    conn = get_db_connection()

    where_clauses = []
    params = []

    if date_from:
        where_clauses.append("timestamp >= ?")
        params.append(date_from)
    if date_to:
        where_clauses.append("timestamp <= ?")
        params.append(date_to)
    if provider:
        where_clauses.append("provider_name = ?")
        params.append(provider)
    if model:
        where_clauses.append("provider_model = ?")
        params.append(model)
    if status:
        where_clauses.append("status = ?")
        params.append(status)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
    SELECT
        run_id,
        timestamp,
        provider_name,
        provider_model,
        input_filename,
        input_pages,
        events_extracted,
        citations_found,
        ROUND(docling_seconds, 2) as docling_seconds,
        ROUND(extractor_seconds, 2) as extractor_seconds,
        ROUND(total_seconds, 2) as total_seconds,
        ROUND(cost_usd, 4) as cost_usd,
        status,
        error_message
    FROM pipeline_runs
    WHERE {where_sql}
    ORDER BY timestamp DESC
    """

    return conn.execute(query, params).df()


# ============================================================================
# UI Helper Functions
# ============================================================================

def render_kpi_card(label: str, value: str, icon: str = "📊", delta: Optional[str] = None):
    """Render a KPI card with gradient background"""
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 2rem;">{icon}</div>
            <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">{label}</div>
            <div style="font-size: 2rem; font-weight: bold; margin-top: 0.25rem;">{value}</div>
            {f'<div style="font-size: 0.85rem; margin-top: 0.25rem; opacity: 0.8;">{delta}</div>' if delta else ''}
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.title("📊 Pipeline Analytics Dashboard")
    st.markdown("Real-time analytics for legal events extraction pipeline")

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")

        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # Global date filter
        st.subheader("📅 Global Filters")
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )

        st.divider()

        # Database info
        st.subheader("💾 Database Info")
        st.caption(f"**Path**: `{DB_PATH.name}`")

        summary = get_summary_stats()
        st.caption(f"**Total Runs**: {summary['total_runs']}")
        st.caption(f"**Last Updated**: {summary['last_run'].strftime('%Y-%m-%d %H:%M') if summary['last_run'] else 'N/A'}")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Summary",
        "⚡ Performance",
        "💰 Cost Tracking",
        "🔍 Run Explorer"
    ])

    # ========================================================================
    # Tab 1: Summary Dashboard
    # ========================================================================
    with tab1:
        st.header("Summary Dashboard")

        summary = get_summary_stats()

        # KPI Cards Row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_kpi_card(
                "Total Runs",
                f"{summary['total_runs']:,}",
                "🚀",
                f"{summary['success_rate']}% success rate"
            )

        with col2:
            render_kpi_card(
                "Total Cost",
                f"${summary['total_cost']:.2f}",
                "💰"
            )

        with col3:
            render_kpi_card(
                "Avg Events/Run",
                f"{summary['avg_events']:.1f}",
                "📄"
            )

        with col4:
            render_kpi_card(
                "Avg Extraction Time",
                f"{summary['avg_extraction_time']:.1f}s",
                "⚡"
            )

        st.divider()

        # Recent Activity
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Recent Runs")
            recent_df = get_recent_runs(limit=10)

            if not recent_df.empty:
                # Format for display
                display_df = recent_df.copy()
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['input_filename'] = display_df['input_filename'].str.slice(0, 30)

                st.dataframe(
                    display_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "status": st.column_config.TextColumn(
                            "Status",
                            help="Run status"
                        ),
                        "cost_usd": st.column_config.NumberColumn(
                            "Cost",
                            format="$%.4f"
                        )
                    }
                )
            else:
                st.info("No runs found in database")

        with col2:
            st.subheader("🏆 Top Models")
            perf_df = get_performance_by_model()

            if not perf_df.empty:
                top_models = perf_df.nlargest(5, 'run_count')[['provider_model', 'run_count']]

                fig = px.bar(
                    top_models,
                    x='run_count',
                    y='provider_model',
                    orientation='h',
                    title="Most Used Models",
                    labels={'run_count': 'Runs', 'provider_model': 'Model'}
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance data available")

    # ========================================================================
    # Tab 2: Performance Analysis
    # ========================================================================
    with tab2:
        st.header("⚡ Performance Analysis")

        perf_df = get_performance_by_model()

        if not perf_df.empty:
            # Speed Comparison Chart
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📊 Extraction Speed by Model")

                fig = px.bar(
                    perf_df.sort_values('avg_extraction_time'),
                    x='avg_extraction_time',
                    y='provider_model',
                    orientation='h',
                    color='provider_name',
                    title="Average Extraction Time (seconds)",
                    labels={'avg_extraction_time': 'Seconds', 'provider_model': 'Model'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("🎯 Performance Metrics")

                # Show top 3 fastest models
                fastest = perf_df.nsmallest(3, 'avg_extraction_time')

                for idx, row in fastest.iterrows():
                    st.metric(
                        label=row['provider_model'][:30],
                        value=f"{row['avg_extraction_time']}s",
                        delta=f"{row['run_count']} runs"
                    )

            st.divider()

            # Quality Metrics Table
            st.subheader("📈 Quality Metrics by Model")

            quality_df = perf_df[['provider_model', 'run_count', 'avg_events', 'avg_citations', 'avg_extraction_time']].copy()
            quality_df.columns = ['Model', 'Runs', 'Avg Events', 'Avg Citations', 'Avg Time (s)']

            st.dataframe(
                quality_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Avg Events": st.column_config.NumberColumn(format="%.1f"),
                    "Avg Citations": st.column_config.NumberColumn(format="%.1f"),
                    "Avg Time (s)": st.column_config.NumberColumn(format="%.2f")
                }
            )

            # Time Series Chart
            st.divider()
            st.subheader("📉 Extraction Time Trends (Last 30 Days)")

            ts_df = get_time_series_data(days=30)

            if not ts_df.empty:
                fig = px.line(
                    ts_df,
                    x='date',
                    y='avg_extraction_time',
                    title="Average Extraction Time Over Time",
                    labels={'date': 'Date', 'avg_extraction_time': 'Avg Time (s)'}
                )
                fig.update_traces(mode='lines+markers')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for time series")

        else:
            st.info("No performance data available. Process some documents first!")

    # ========================================================================
    # Tab 3: Cost Tracking
    # ========================================================================
    with tab3:
        st.header("💰 Cost Tracking")

        cost_df = get_cost_breakdown()

        if not cost_df.empty:
            # Total spending summary
            total_spending = cost_df['total_cost'].sum()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("💵 Total Spending", f"${total_spending:.2f}")

            with col2:
                avg_per_run = cost_df['avg_cost_per_run'].mean()
                st.metric("📊 Avg Cost/Run", f"${avg_per_run:.4f}")

            with col3:
                total_events = cost_df['total_events'].sum()
                cost_per_event = total_spending / total_events if total_events > 0 else 0
                st.metric("📄 Cost/Event", f"${cost_per_event:.4f}")

            st.divider()

            # Spending breakdown charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🥧 Spending by Provider")

                provider_costs = cost_df.groupby('provider_name')['total_cost'].sum().reset_index()

                fig = px.pie(
                    provider_costs,
                    values='total_cost',
                    names='provider_name',
                    title="Cost Distribution by Provider"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 Cost Efficiency by Model")

                efficiency_df = cost_df.nsmallest(10, 'cost_per_event')[['provider_model', 'cost_per_event']].copy()

                fig = px.bar(
                    efficiency_df,
                    x='cost_per_event',
                    y='provider_model',
                    orientation='h',
                    title="Cost per Event Extracted",
                    labels={'cost_per_event': 'Cost ($)', 'provider_model': 'Model'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Detailed cost table
            st.subheader("📋 Detailed Cost Breakdown")

            display_cost_df = cost_df.copy()
            display_cost_df.columns = [
                'Provider', 'Model', 'Runs', 'Total Cost',
                'Avg Cost/Run', 'Cost/Event', 'Total Events'
            ]

            st.dataframe(
                display_cost_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Total Cost": st.column_config.NumberColumn(format="$%.4f"),
                    "Avg Cost/Run": st.column_config.NumberColumn(format="$%.4f"),
                    "Cost/Event": st.column_config.NumberColumn(format="$%.4f")
                }
            )

            # Spending timeline
            st.divider()
            st.subheader("📈 Cumulative Spending Over Time")

            ts_df = get_time_series_data(days=30)

            if not ts_df.empty:
                ts_df['cumulative_cost'] = ts_df['daily_cost'].cumsum()

                fig = px.area(
                    ts_df,
                    x='date',
                    y='cumulative_cost',
                    title="Cumulative Cost (Last 30 Days)",
                    labels={'date': 'Date', 'cumulative_cost': 'Total Cost ($)'}
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No cost data available. Ensure your runs include cost metadata!")

    # ========================================================================
    # Tab 4: Run Explorer
    # ========================================================================
    with tab4:
        st.header("🔍 Run Explorer")

        # Filters
        col1, col2, col3, col4 = st.columns(4)

        # Get unique values for filters
        conn = get_db_connection()
        providers = conn.execute("SELECT DISTINCT provider_name FROM pipeline_runs ORDER BY provider_name").fetchall()
        models = conn.execute("SELECT DISTINCT provider_model FROM pipeline_runs ORDER BY provider_model").fetchall()
        statuses = conn.execute("SELECT DISTINCT status FROM pipeline_runs ORDER BY status").fetchall()

        with col1:
            filter_provider = st.selectbox(
                "Provider",
                options=['All'] + [p[0] for p in providers]
            )

        with col2:
            filter_model = st.selectbox(
                "Model",
                options=['All'] + [m[0] for m in models]
            )

        with col3:
            filter_status = st.selectbox(
                "Status",
                options=['All'] + [s[0] for s in statuses]
            )

        with col4:
            st.write("")  # Spacing
            st.write("")
            export_csv = st.button("📥 Export to CSV", use_container_width=True)

        # Date filter
        col1, col2 = st.columns(2)
        with col1:
            filter_date_from = st.date_input("From Date", value=None)
        with col2:
            filter_date_to = st.date_input("To Date", value=None)

        # Apply filters
        filtered_df = get_filtered_runs(
            date_from=datetime.combine(filter_date_from, datetime.min.time()) if filter_date_from else None,
            date_to=datetime.combine(filter_date_to, datetime.max.time()) if filter_date_to else None,
            provider=filter_provider if filter_provider != 'All' else None,
            model=filter_model if filter_model != 'All' else None,
            status=filter_status if filter_status != 'All' else None
        )

        st.divider()

        # Results
        st.subheader(f"📊 Results ({len(filtered_df)} runs)")

        if not filtered_df.empty:
            # Format for display
            display_df = filtered_df.copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                height=400,
                column_config={
                    "run_id": st.column_config.TextColumn("Run ID", width="medium"),
                    "cost_usd": st.column_config.NumberColumn("Cost", format="$%.4f"),
                    "status": st.column_config.TextColumn("Status", width="small")
                }
            )

            # Export functionality
            if export_csv:
                csv = filtered_df.to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"pipeline_runs_{timestamp}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No runs match the selected filters")


if __name__ == "__main__":
    main()
