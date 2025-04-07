import streamlit as st

# fix for torch classes not found error
import torch
torch.classes.__path__ = []

from mcq_generator import MCQGenerator

# Initialize session state variables
if 'mcq_generator' not in st.session_state:
    st.session_state.mcq_generator = None
if 'session_started' not in st.session_state:
    st.session_state.session_started = False
if 'answer_submitted' not in st.session_state:
    st.session_state.answer_submitted = False
if 'session_end' not in st.session_state:
    st.session_state.session_end = False

# Function to load a new question
def load_new_question():
    _ = st.session_state.mcq_generator.generate_mcq()
    st.session_state.answer_submitted = False

# Landing page: context input
if not st.session_state.session_started and not st.session_state.session_end:
    st.header("MCQ Generator - Setup")
    context_text = st.text_area("Enter the context for MCQ questions:")
    if st.button("Submit Context") and context_text.strip():
        st.session_state.mcq_generator = MCQGenerator()
        st.session_state.mcq_generator.process_user_query(context_text)
        st.session_state.session_started = True
        load_new_question()
        st.rerun()

# Main MCQ session
if st.session_state.session_started:
    # get current question
    cur_mcq = st.session_state.mcq_generator.cur_mcq

    st.header("MCQ Question")
    st.write(cur_mcq["question"])
    # Display options as radio buttons if answer not submitted
    if not st.session_state.answer_submitted:
        # ask quesion
        user_answer = st.radio("Select an answer:", cur_mcq['choices'], index=None)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Submit Answer", disabled=user_answer is None):
                _ = st.session_state.mcq_generator.answer_mcq(user_answer)
                st.session_state.answer_submitted = True
                st.rerun()
        with col2:
            if st.button("Next Question"):
                st.session_state.mcq_generator.reject_mcq()
                load_new_question()
                st.rerun()
        with col3:
            if st.button("End MCQ"):
                st.session_state.session_started = False
                st.session_state.session_end = True
                st.rerun()
    else:
        # question has been answered
        # generate feedback
        if cur_mcq['answer_is_correct']:
            feedback = f"Correct! \n\nExplanation: {cur_mcq['explanation']}"
        else:
            feedback = f"Wrong! \n\nExplanation: {cur_mcq['explanation']}"
        st.write(feedback)
    
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Next Question"):
                load_new_question()
                st.rerun()
        with col2:
            if st.button("End MCQ"):
                st.session_state.session_started = False
                st.session_state.session_end = True
                st.rerun()

# End of MCQ session
if st.session_state.session_end:
    result_df = st.session_state.mcq_generator.get_result_df()
    st.write("Session Ended. Here are your results:")
    st.bar_chart(result_df)

    if st.button("Start New Session"):
        st.session_state.session_started = False
        st.session_state.session_end = False
        st.session_state.mcq_generator = None
        st.session_state.answer_submitted = False
        st.rerun()
