import streamlit as st

def render_teaching_explanation(title: str, what: str, why: str, how: str):
    """
    Renders an expandable educational component explaining the concepts.
    """
    with st.expander(f"🎓 Teaching Mode: {title}"):
        st.markdown(f"**WHAT?** {what}")
        st.markdown(f"**WHY?** {why}")
        st.markdown(f"**HOW?** {how}")

def render_page_header(title: str, description: str):
    """
    Renders a consistent header for each page.
    """
    st.markdown(f"# {title}")
    st.markdown(f"_{description}_")
    st.markdown("---")
