"""
Streamlit Web UI for HCP Merge Assistant

A user-friendly interface for Data Stewards to:
1. Upload HCP files for analysis
2. Review match results with visual comparisons
3. Select/deselect merges
4. Execute bulk merges with progress tracking
5. Download reports
"""

import streamlit as st
import pandas as pd
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.merge_assistant import MergeAssistant, AssistantConfig, AnalysisSession
from src.core.match_analyzer import MatchConfidence


# Page configuration
st.set_page_config(
    page_title="HCP Merge Assistant",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .match-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f9f9f9;
    }
    .match-score-high { color: #28a745; font-weight: bold; }
    .match-score-medium { color: #ffc107; font-weight: bold; }
    .match-score-low { color: #dc3545; font-weight: bold; }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    if "session" not in st.session_state:
        st.session_state.session = None
    if "step" not in st.session_state:
        st.session_state.step = "config"
    if "selected_merges" not in st.session_state:
        st.session_state.selected_merges = set()


def render_sidebar():
    """Render the sidebar with configuration"""
    with st.sidebar:
        st.image("https://www.reltio.com/wp-content/uploads/2021/08/reltio-logo.svg", width=150)
        st.markdown("---")

        st.subheader("🔧 Configuration")

        # Environment selection
        environment = st.selectbox(
            "Reltio Environment",
            ["dev", "test", "prod", "prod-usg"],
            index=0
        )

        # Credentials
        with st.expander("Reltio Credentials", expanded=True):
            client_id = st.text_input(
                "Client ID",
                value=os.getenv("RELTIO_CLIENT_ID", ""),
                type="password"
            )
            client_secret = st.text_input(
                "Client Secret",
                value=os.getenv("RELTIO_CLIENT_SECRET", ""),
                type="password"
            )
            tenant_id = st.text_input(
                "Tenant ID",
                value=os.getenv("RELTIO_TENANT_ID", "")
            )

        # LLM Configuration
        with st.expander("LLM Configuration (Optional)", expanded=False):
            llm_provider = st.radio(
                "Provider",
                ["OpenAI", "Anthropic", "None"],
                horizontal=True
            )
            llm_api_key = st.text_input(
                "API Key",
                value=os.getenv("OPENAI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", ""),
                type="password"
            )

        # Entity type
        entity_type = st.selectbox(
            "Entity Type",
            ["HCP", "HCO", "Product", "Individual", "Organization"],
            index=0
        )

        # Thresholds
        with st.expander("Match Thresholds", expanded=False):
            auto_merge_threshold = st.slider(
                "Auto-merge threshold (%)",
                min_value=80,
                max_value=100,
                value=95
            )
            review_threshold = st.slider(
                "Review threshold (%)",
                min_value=50,
                max_value=90,
                value=70
            )

        st.markdown("---")

        # Initialize button
        if st.button("🚀 Initialize Assistant", use_container_width=True):
            try:
                config = AssistantConfig(
                    reltio_client_id=client_id,
                    reltio_client_secret=client_secret,
                    reltio_tenant_id=tenant_id,
                    reltio_environment=environment,
                    llm_api_key=llm_api_key if llm_provider != "None" else None,
                    llm_provider=llm_provider.lower() if llm_provider != "None" else "openai",
                    use_llm=llm_provider != "None" and bool(llm_api_key),
                    entity_type=entity_type,
                    auto_merge_threshold=auto_merge_threshold,
                    review_threshold=review_threshold
                )
                st.session_state.assistant = MergeAssistant(config)
                st.session_state.step = "upload"
                st.success("✅ Assistant initialized!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Initialization failed: {e}")

        # Health check
        if st.session_state.assistant:
            if st.button("🏥 Health Check", use_container_width=True):
                with st.spinner("Checking connections..."):
                    health = st.session_state.assistant.run_health_check()
                    if health["reltio"]["status"] == "healthy":
                        st.success("✅ Reltio: Connected")
                    else:
                        st.error(f"❌ Reltio: {health['reltio'].get('error', 'Unknown error')}")

                    if health["llm"]["status"] == "configured":
                        st.success(f"✅ LLM: {health['llm']['provider']}")
                    elif health["llm"]["status"] == "not_configured":
                        st.info("ℹ️ LLM: Not configured")

        return {
            "environment": environment,
            "client_id": client_id,
            "client_secret": client_secret,
            "tenant_id": tenant_id,
            "entity_type": entity_type
        }


def render_upload_step():
    """Render the file upload step"""
    st.markdown('<p class="main-header">📁 Upload HCP File</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a CSV, Excel, or JSON file with HCP records to analyze</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "json"],
            help="Supported formats: CSV, Excel (.xlsx, .xls), JSON"
        )

        if uploaded_file:
            # Preview the file
            st.subheader("📋 File Preview")

            # Save to temp file for parsing
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                # Parse file
                parsed = st.session_state.assistant.parse_file(tmp_path)

                # Show preview
                preview_data = []
                for record in parsed.records[:10]:
                    row = {"Row": record.row_number}
                    row.update(record.normalized_data)
                    preview_data.append(row)

                df = pd.DataFrame(preview_data)
                st.dataframe(df, use_container_width=True)

                # Column mapping
                st.subheader("🔄 Column Mapping")
                mapping_df = pd.DataFrame([
                    {"Original Column": k, "Mapped To": v}
                    for k, v in parsed.column_mapping.items()
                ])
                st.dataframe(mapping_df, use_container_width=True)

                # Summary metrics
                st.subheader("📊 File Summary")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Records", parsed.total_records)
                col_b.metric("Valid Records", parsed.valid_records)
                col_c.metric("Invalid Records", parsed.invalid_records)

                # Create session and proceed
                if st.button("🔍 Analyze Matches", type="primary", use_container_width=True):
                    session = st.session_state.assistant.create_session(tmp_path)
                    st.session_state.session = session
                    st.session_state.step = "analyze"
                    st.rerun()

            except Exception as e:
                st.error(f"Error parsing file: {e}")
            finally:
                # Cleanup temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    with col2:
        st.info("""
        **📌 File Requirements**

        Your file should contain columns like:
        - NPI (National Provider Identifier)
        - First Name, Last Name
        - Specialty
        - Address, City, State, ZIP
        - Phone, Email

        **💡 Tip:** Include `entity_id` or `merge_target` column if you know which Reltio entities to merge.
        """)

        # Sample file download
        st.subheader("📥 Sample Files")
        sample_csv = """NPI,FirstName,LastName,Specialty,City,State
1234567890,John,Smith,Oncology,Boston,MA
0987654321,Jane,Doe,Cardiology,New York,NY
"""
        st.download_button(
            "Download Sample CSV",
            sample_csv,
            "sample_hcp.csv",
            "text/csv",
            use_container_width=True
        )


def render_analysis_step():
    """Render the analysis step with progress"""
    st.markdown('<p class="main-header">🔍 Analyzing Matches</p>', unsafe_allow_html=True)

    session = st.session_state.session

    if not session.match_results:
        st.info("Running match analysis against Reltio...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(completed, total):
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Analyzing record {completed} of {total}...")

        try:
            results = st.session_state.assistant.run_analysis(session, on_progress)
            st.session_state.step = "review"
            st.rerun()
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            if st.button("🔄 Retry"):
                st.rerun()
    else:
        st.session_state.step = "review"
        st.rerun()


def render_review_step():
    """Render the match review step"""
    st.markdown('<p class="main-header">📊 Review Matches</p>', unsafe_allow_html=True)

    session = st.session_state.session
    results = session.match_results
    summary = st.session_state.assistant.get_merge_summary(session)

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", summary["total_records"])
    col2.metric("With Matches", summary["with_matches"])
    col3.metric("No Matches", summary["no_matches"])
    col4.metric("Selected for Merge", len(st.session_state.selected_merges))
    col5.metric("Exact Matches", summary["by_confidence"]["exact"])

    st.markdown("---")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_confidence = st.multiselect(
            "Filter by Confidence",
            ["exact", "high", "medium", "low", "uncertain"],
            default=["exact", "high", "medium"]
        )
    with col_f2:
        filter_recommendation = st.multiselect(
            "Filter by Recommendation",
            ["merge", "review", "skip", "no_match"],
            default=["merge", "review"]
        )
    with col_f3:
        show_only_selected = st.checkbox("Show only selected")

    # Bulk actions
    st.subheader("⚡ Bulk Actions")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)

    with col_a1:
        if st.button("✅ Select All High Confidence", use_container_width=True):
            for i, r in enumerate(results):
                if r.best_match and r.best_match.confidence in [MatchConfidence.EXACT, MatchConfidence.HIGH]:
                    st.session_state.selected_merges.add(i)
            st.rerun()

    with col_a2:
        if st.button("🔘 Select Exact Matches Only", use_container_width=True):
            for i, r in enumerate(results):
                if r.best_match and r.best_match.confidence == MatchConfidence.EXACT:
                    st.session_state.selected_merges.add(i)
            st.rerun()

    with col_a3:
        if st.button("❌ Clear All Selections", use_container_width=True):
            st.session_state.selected_merges.clear()
            st.rerun()

    with col_a4:
        if st.button("🔄 Auto-Select (AI)", use_container_width=True):
            count = st.session_state.assistant.auto_select_merges(session)
            for i in session.selected_merges:
                st.session_state.selected_merges.add(i)
            st.success(f"Auto-selected {count} merges")
            st.rerun()

    st.markdown("---")

    # Match results table
    st.subheader("📋 Match Results")

    # Filter results
    filtered_results = []
    for i, result in enumerate(results):
        confidence = result.best_match.confidence.value if result.best_match else "no_match"
        recommendation = result.recommendation

        if confidence not in filter_confidence and "no_match" not in filter_confidence:
            continue
        if recommendation not in filter_recommendation:
            continue
        if show_only_selected and i not in st.session_state.selected_merges:
            continue

        filtered_results.append((i, result))

    # Pagination
    items_per_page = 20
    total_pages = max(1, (len(filtered_results) + items_per_page - 1) // items_per_page)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_results = filtered_results[start_idx:end_idx]

    st.text(f"Showing {start_idx + 1}-{min(end_idx, len(filtered_results))} of {len(filtered_results)} results")

    # Render each result
    for idx, result in page_results:
        render_match_result(idx, result)

    st.markdown("---")

    # Proceed to merge
    if st.session_state.selected_merges:
        st.success(f"✅ {len(st.session_state.selected_merges)} records selected for merge")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("🧪 Dry Run (Validate)", type="secondary", use_container_width=True):
                st.session_state.step = "merge"
                st.session_state.dry_run = True
                st.rerun()

        with col_m2:
            if st.button("🔀 Execute Merges", type="primary", use_container_width=True):
                st.session_state.step = "merge"
                st.session_state.dry_run = False
                st.rerun()
    else:
        st.warning("⚠️ No records selected for merge. Select at least one record to proceed.")


def render_match_result(idx: int, result):
    """Render a single match result card"""
    input_data = result.input_record.normalized_data
    best_match = result.best_match
    is_selected = idx in st.session_state.selected_merges

    # Determine color based on confidence
    if best_match:
        if best_match.confidence == MatchConfidence.EXACT:
            border_color = "#28a745"
        elif best_match.confidence == MatchConfidence.HIGH:
            border_color = "#17a2b8"
        elif best_match.confidence == MatchConfidence.MEDIUM:
            border_color = "#ffc107"
        else:
            border_color = "#dc3545"
    else:
        border_color = "#6c757d"

    with st.container():
        st.markdown(f"""
        <div style="border-left: 4px solid {border_color}; padding-left: 1rem; margin: 1rem 0;">
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 2])

        with col1:
            # Selection checkbox
            selected = st.checkbox(
                f"Row {result.input_record.row_number}",
                value=is_selected,
                key=f"select_{idx}"
            )
            if selected and idx not in st.session_state.selected_merges:
                st.session_state.selected_merges.add(idx)
            elif not selected and idx in st.session_state.selected_merges:
                st.session_state.selected_merges.discard(idx)

            # Score and confidence
            if best_match:
                score_class = "high" if best_match.match_score >= 90 else ("medium" if best_match.match_score >= 70 else "low")
                st.markdown(f'<span class="match-score-{score_class}">Score: {best_match.match_score:.0f}%</span>', unsafe_allow_html=True)
                st.caption(f"Confidence: {best_match.confidence.value}")
                st.caption(f"Rec: {result.recommendation}")
            else:
                st.caption("No matches found")

        with col2:
            st.markdown("**📄 Input Record**")
            for key, value in list(input_data.items())[:6]:
                st.text(f"{key}: {value}")

        with col3:
            if best_match:
                st.markdown("**🎯 Best Match (Reltio)**")
                st.text(f"URI: {best_match.entity_uri}")
                st.text(f"Label: {best_match.entity_label}")

                # Show matched attributes
                if best_match.matched_attributes:
                    with st.expander("Matched Attributes"):
                        for attr, (input_val, reltio_val) in best_match.matched_attributes.items():
                            st.text(f"{attr}: {input_val} ↔ {reltio_val}")

                # Show other candidates
                if len(result.candidates) > 1:
                    with st.expander(f"Other Candidates ({len(result.candidates) - 1})"):
                        for i, cand in enumerate(result.candidates[1:5], 1):
                            st.text(f"{i}. {cand.entity_label} ({cand.match_score:.0f}%)")
            else:
                st.markdown("**No matching entity found in Reltio**")

        # LLM Analysis
        if result.llm_analysis:
            with st.expander("🤖 AI Analysis"):
                st.text(result.llm_analysis)

        st.markdown("</div>", unsafe_allow_html=True)


def render_merge_step():
    """Render the merge execution step"""
    st.markdown('<p class="main-header">🔀 Executing Merges</p>', unsafe_allow_html=True)

    session = st.session_state.session
    dry_run = getattr(st.session_state, 'dry_run', False)

    # Update selected merges in session
    for idx in st.session_state.selected_merges:
        session.selected_merges[idx] = 0  # Use best match

    st.info(f"{'🧪 DRY RUN MODE - Validating' if dry_run else '🔀 Executing'} {len(st.session_state.selected_merges)} merge operations...")

    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    def on_progress(completed, total, operation):
        progress = completed / total
        progress_bar.progress(progress)
        status_text.text(f"Processing {completed}/{total}: {operation.winner_uri} ← {operation.loser_uri}")

    try:
        result = st.session_state.assistant.run_merges(session, on_progress, dry_run=dry_run)

        st.success(f"✅ Completed! {result.successful} successful, {result.failed} failed")

        # Results summary
        with results_container:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", result.total_operations)
            col2.metric("Successful", result.successful)
            col3.metric("Failed", result.failed)
            col4.metric("Duration", f"{result.duration_seconds:.1f}s")

            if result.errors:
                with st.expander("❌ Errors"):
                    for error in result.errors[:10]:
                        st.error(error)

            # Download report
            report = st.session_state.assistant.generate_report(session)
            report_json = json.dumps(report, indent=2, default=str)

            st.download_button(
                "📥 Download Report (JSON)",
                report_json,
                f"merge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )

        st.session_state.step = "complete"

    except Exception as e:
        st.error(f"Merge execution failed: {e}")
        if st.button("🔄 Retry"):
            st.rerun()

    if st.button("← Back to Review"):
        st.session_state.step = "review"
        st.rerun()


def render_complete_step():
    """Render the completion step"""
    st.markdown('<p class="main-header">✅ Merge Complete</p>', unsafe_allow_html=True)

    session = st.session_state.session

    st.balloons()

    if session.merge_result:
        st.success(f"""
        **Merge Summary**
        - Total Operations: {session.merge_result.total_operations}
        - Successful: {session.merge_result.successful}
        - Failed: {session.merge_result.failed}
        - Duration: {session.merge_result.duration_seconds:.2f} seconds
        """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📁 Start New Session", type="primary", use_container_width=True):
            st.session_state.session = None
            st.session_state.selected_merges = set()
            st.session_state.step = "upload"
            st.rerun()

    with col2:
        report = st.session_state.assistant.generate_report(session)
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            "📥 Download Full Report",
            report_json,
            f"merge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )


def main():
    """Main application entry point"""
    init_session_state()

    # Sidebar
    config = render_sidebar()

    # Main content based on step
    if not st.session_state.assistant:
        st.markdown('<p class="main-header">🔗 HCP Merge Assistant</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Intelligent bulk merge solution for Reltio MDM</p>', unsafe_allow_html=True)

        st.info("👈 Configure your Reltio credentials in the sidebar and click **Initialize Assistant** to begin.")

        st.markdown("""
        ### Features
        - 📁 **Upload** CSV, Excel, or JSON files with HCP data
        - 🔍 **Analyze** matches using Reltio's matching engine
        - 🤖 **AI-Powered** recommendations using GPT-4 or Claude
        - 📊 **Review** matches with visual comparisons
        - 🔀 **Bulk Merge** with progress tracking
        - 📥 **Export** detailed reports

        ### Workflow
        1. Configure Reltio credentials
        2. Upload HCP file
        3. Review match results
        4. Select records to merge
        5. Execute merges
        6. Download report
        """)

    elif st.session_state.step == "upload":
        render_upload_step()

    elif st.session_state.step == "analyze":
        render_analysis_step()

    elif st.session_state.step == "review":
        render_review_step()

    elif st.session_state.step == "merge":
        render_merge_step()

    elif st.session_state.step == "complete":
        render_complete_step()


if __name__ == "__main__":
    main()
