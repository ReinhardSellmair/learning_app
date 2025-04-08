# class to generate multiple choice questions
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import random
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from config import OPEN_AI_MODEL, TEMPERATURE, EMBEDDING, N_KEYWORDS, N_WIKI_PAGES, PAGE_SIM_COL, MIN_PAGE_SIMILARITY, \
    MIN_PAGES_KEEP, MIN_SECTION_SIMILARITY, MIN_SECTIONS_KEEP, REJECTION_WEIGHT, N_SECTIONS_SELECT, N_CHOICES, \
    FAILED_QUERY, MAX_RETRIES
from prompts import keyword_sys_prompt, question_generation_prompt
from wiki_search import get_wiki_page_info_df
from wiki_api import get_wiki_page_sections_as_dict

class MCQGenerator:

    def __init__(self, chat_model_name=OPEN_AI_MODEL, temperature=TEMPERATURE, embedding_model_name=EMBEDDING):
        """
        Initializes the MCQGenerator class with the specified chat model, temperature, and embedding model.

        Args:
            chat_model_name (str): Name of the chat model to use.
            temperature (float): Temperature setting for the chat model.
            embedding_model_name (str): Name of the embedding model to use.
        """        
        logging.info(f"Initializing MCQGenerator with chat model: {chat_model_name}, temperature: {temperature}, embedding model: {embedding_model_name}")
        # initialize chat model
        self.chat_model = ChatOpenAI(model=chat_model_name, temperature=temperature)
        # initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        # initialze current mcq
        self.cur_mcq = None

    def process_user_query(self, user_query, n_keywords=N_KEYWORDS, number_of_results=N_WIKI_PAGES, 
                           page_sim_col=PAGE_SIM_COL, min_page_similarity=MIN_PAGE_SIMILARITY, 
                           min_pages_keep=MIN_PAGES_KEEP, min_section_similarity=MIN_SECTION_SIMILARITY,
                           min_sections_keep=MIN_SECTIONS_KEEP, rejection_weight=REJECTION_WEIGHT):
        """
        Processes the user query to generate keywords, retrieve wiki pages, and filter sections.

        Args:
            user_query (str): The user's query.
            n_keywords (int): Number of keywords to extract.
            number_of_results (int): Number of wiki pages to retrieve.
            page_sim_col (str): Column name for page similarity.
            min_page_similarity (float): Minimum similarity threshold for pages.
            min_pages_keep (int): Minimum number of pages to keep.
            min_section_similarity (float): Minimum similarity threshold for sections.
            min_sections_keep (int): Minimum number of sections to keep.
            rejection_weight (float): Weight for rejection scores.
        """        
        logging.info(f"Processing user query: \"{user_query}\"")
        # log all input parameters
        logging.info(f"Input parameters: n_keywords={n_keywords}, number_of_results={number_of_results}, "
                     f"page_sim_col={page_sim_col}, min_page_similarity={min_page_similarity}, "
                     f"min_pages_keep={min_pages_keep}, min_section_similarity={min_section_similarity}, "
                     f"min_sections_keep={min_sections_keep}, rejection_weight={rejection_weight}")
        
        # set rejection weight
        self.rejection_weight = rejection_weight

        # extract content for user query
        self.user_query = user_query
        # embed user query
        self.user_query_embedding = self.embedding_model.encode(user_query)
        
        # get keywords from chat model
        logging.info(f"Getting keywords for user query ...")
        keywords = self._get_wiki_keywords(n_keywords)
        logging.info(f"Retrieved keywords: {keywords}")
        
        # get wiki page information for keywords
        logging.info(f"Getting wiki page information for keywords ...")
        self.pages_df = get_wiki_page_info_df(keywords, number_of_results)
        logging.info(f"Retrieved {len(self.pages_df)} wiki pages")

        # add page similarity
        logging.info(f"Adding page similarity ...")
        self._filter_pages(page_sim_col, min_page_similarity, min_pages_keep)
        logging.info(f"Number of pages after filtering: {len(self.pages_df)}")

        # read page sections
        logging.info(f"Reading page sections ...")
        self._get_page_sections_df()
        logging.info(f"Retrieved {len(self.sections_df)} sections")

        # filter sections by similarity
        logging.info(f"Filtering sections by similarity ...")
        self._filter_sections(min_section_similarity, min_sections_keep)
        logging.info(f"Number of sections after filtering: {len(self.sections_df)}")

        # set score
        self._set_score()

        # initialize questions
        self.questions_df = pd.DataFrame()

    def generate_mcq(self, n_sections_select=N_SECTIONS_SELECT, n_choices=N_CHOICES, max_retries=MAX_RETRIES):
        """
        Generates a multiple-choice question based on the filtered sections.

        Args:
            n_sections_select (int): Number of sections to select for generating the question.
            n_choices (int): Number of answer choices for the question.

        Returns:
            dict: The generated multiple-choice question.
        """        
        def get_mcq():
            # select sections by score
            sections_select_df = (self.sections_df.sample(n_sections_select, weights=self.sections_df['score'])
                                .sort_values('score', ascending=False))
            logging.info(f"selected sections: {sections_select_df['section'].tolist()}")

            # generate context
            apply_fcn = lambda row: f"Title: {row['title']}, Section: {row['section']}\n{row['text']}"
            context_text = '\n\n'.join(sections_select_df.apply(apply_fcn, axis=1))

            # generate question prompt
            question_prompt = question_generation_prompt(self.user_query, context_text, self.questions_df, n_choices)

            # query chat model
            logging.info(f"Querying chat model ...")
            messages = [SystemMessage(content=question_prompt)]
            response = self.chat_model.invoke(messages)
            self.cur_mcq = json.loads(response.content)

            # add section ids to mcq
            self.cur_mcq['section_ids'] = sections_select_df.index.tolist()


        logging.info(f"Generating multiple choice question ...")
        logging.info(f"generate_mcq parameters: n_sections_select={n_sections_select}, n_choices={n_choices}, max_retries={max_retries}")

        get_mcq()

        # check if question generation failed
        retries = 0
        while retries < max_retries and self.cur_mcq['question'] == FAILED_QUERY:
            logging.info(f"Question generation failed. Retrying ...")
            
            # update rejection count
            self._update_rejection_count()

            # re-generate question
            get_mcq()

            retries += 1

        # check if question could be generated
        if self.cur_mcq['question'] == FAILED_QUERY:
            logging.error(f"Failed to generate question after {max_retries} retries.")
            return None

        logging.info(f"Generated question: {self.cur_mcq['question']}")

        # get correct answer
        self.cur_mcq['correct_answer'] = self.cur_mcq['choices'][0]

        # shuffle choices
        random.shuffle(self.cur_mcq['choices'])        

        return self.cur_mcq
    
    def reject_mcq(self):
        """
        Handles the rejection of a generated multiple-choice question by the user.
        Updates rejection counts for sections and pages and recalculates scores.
        """        
        logging.info(f"User rejected question")
        # add questions
        self._add_question(None, question_rejected=True)

        # update rejection count
        self._update_rejection_count()

    def answer_mcq(self, user_answer):
        """
        Processes the user's answer to the current multiple-choice question.

        Args:
            user_answer (str): The user's answer to the question.
        """        
        # add question to questions dataframe
        self._add_question(user_answer, question_rejected=False)

        self.cur_mcq['user_answer'] = user_answer

        # check if answer is correct
        self.cur_mcq['answer_is_correct'] = self.cur_mcq['correct_answer'] == user_answer
        logging.info(f"User answer: \"{user_answer}\" is {'correct' if self.cur_mcq['answer_is_correct'] else 'incorrect'}")

        return self.cur_mcq['answer_is_correct']
    
    def get_result_df(self):
        """
        Returns the results of the MCQ generation session as a DataFrame.

        Returns:
            pd.DataFrame: The results of the MCQ generation session.
        """        
        questions_df = self.questions_df.copy()

        # get number of rejected questions
        rejected_questions = questions_df['question_rejected'].sum()

        # get number of correct and incorrect answers
        count = questions_df.loc[~questions_df['question_rejected'], 'is_correct'].value_counts()
        correct_answers = count.get(True, 0)
        incorrect_answers = count.get(False, 0)
        # get number of correct answers
        correct_answers = self.questions_df['is_correct'].sum()
        # get number of rejected questions
        rejected_questions = self.questions_df['question_rejected'].sum()

        results = {
            'Correct Answers': correct_answers,
            'Wrong Answers': incorrect_answers,
            'Rejected Questions': rejected_questions
        }

        return pd.DataFrame.from_dict(results, orient='index', columns=["Count"])
    
    def _update_rejection_count(self):
        # get selected section ids
        section_ids = self.cur_mcq['section_ids']
        # update section rejection count
        self.sections_df.loc[section_ids, 'n_rejected'] += 1
        # get rejected titles
        rejected_titles = self.sections_df.loc[section_ids, 'title'].unique()
        # update page rejection count
        self.pages_df.loc[self.pages_df['title'].isin(rejected_titles), 'n_rejected'] += 1
        # update section scores
        self._set_score()

        
    def _get_wiki_keywords(self, n_keywords):
        """
        Retrieves keywords for the user query using the chat model.

        Args:
            n_keywords (int): Number of keywords to extract.

        Returns:
            list: A list of extracted keywords.
        """        
        # get keywords for user query
        messages = [SystemMessage(content=keyword_sys_prompt(n_keywords)), HumanMessage(content=self.user_query)]
        response = self.chat_model.invoke(messages)
        return response.content.split(', ')
    
    def _filter_pages(self, embedding_col=PAGE_SIM_COL, min_similarity=MIN_PAGE_SIMILARITY, min_keep=MIN_PAGES_KEEP):
        """
        Filters wiki pages based on similarity to the user query.

        Args:
            embedding_col (str): Column name for embeddings.
            min_similarity (float): Minimum similarity threshold for filtering.
            min_keep (int): Minimum number of pages to keep.
        """        
        # embed pages
        self.pages_df['embedding'] = self.pages_df[embedding_col].fillna('').map(self.embedding_model.encode)
        # get similarity
        self.pages_df['page_similarity'] = cosine_similarity([self.user_query_embedding], 
                                                              self.pages_df['embedding'].tolist())[0]
        
        # filter pages by context similarity
        self.pages_df = self._filter_df_by_similarity(self.pages_df, 'page_similarity', min_similarity, min_keep)

    def _get_page_sections_df(self):
        """
        Retrieves sections for each wiki page and stores them in a DataFrame.
        """        
        # read text of all sections for each page
        sections = []
        # iterate through pages
        for title in self.pages_df['title']:
            wiki_doc = get_wiki_page_sections_as_dict(title)
            # iterate through sections
            for section, text in wiki_doc.items():
                sections.append({'title': title, 'section': section, 'text': text, 'n_rejected': 0})

        self.sections_df = pd.DataFrame(sections)

    def _filter_sections(self, min_similarity=MIN_SECTION_SIMILARITY, min_keep=MIN_SECTIONS_KEEP):
        """
        Filters sections based on similarity to the user query.

        Args:
            min_similarity (float): Minimum similarity threshold for filtering.
            min_keep (int): Minimum number of sections to keep.
        """        
        # emmbed sections
        self.sections_df['embedding'] = (self.sections_df.apply(lambda row: f"{row['section']}: {row['text']}", axis=1)
                                         .map(self.embedding_model.encode))
        # get similarity
        self.sections_df['section_similarity'] = cosine_similarity([self.user_query_embedding], 
                                                                    self.sections_df['embedding'].tolist())[0]
        
        self.sections_df = self._filter_df_by_similarity(self.sections_df, 'section_similarity', min_similarity, 
                                                         min_keep)
    def _set_score(self):
        """
        Calculates and sets scores for sections based on similarity and rejection counts.
        """         
        # normalize rejection counts
        self.pages_df['rejection_score'] = ((1 - self.pages_df['n_rejected'] / self.pages_df['n_rejected'].max())
                                            .fillna(1))
        self.sections_df['rejection_score'] = ((1 - self.sections_df['n_rejected'] / self.sections_df['n_rejected']
                                                .max()).fillna(1))
        # join number of page rejections
        self.sections_df = (self.sections_df.drop(columns='page_rejection_score', errors='ignore')
                            .merge(self.pages_df[['title', 'rejection_score']]
                                   .rename(columns={'rejection_score': 'page_rejection_score'}), on='title', how='left'))
        
        # combine page and section rejections
        self.sections_df['rejection_score_combined'] = (self.sections_df['rejection_score'] + 
                                                        self.sections_df['page_rejection_score']) / 2
        # calculate score
        self.sections_df['score'] = ((1 - self.rejection_weight) * self.sections_df['section_similarity'] + 
                                      self.rejection_weight * self.sections_df['rejection_score_combined'])

    def _add_question(self, user_answer, question_rejected):
        """
        Adds the current question to the questions DataFrame.

        Args:
            user_answer (str): The user's answer to the question.
            question_rejected (bool): Whether the question was rejected by the user.
        """        
        # add question to questions dataframe
        new_question = self.cur_mcq.copy()
        # get correct answer
        new_question['correct_answer'] = self.cur_mcq['correct_answer']
        # add user answer
        new_question['user_answer'] = user_answer
        # check if user answer is correct
        new_question['is_correct'] = new_question['correct_answer'] == user_answer
        # add question rejection
        new_question['question_rejected'] = question_rejected
        
        # add to dataframe
        self.questions_df = pd.concat([self.questions_df, pd.DataFrame([new_question])], ignore_index=True)        
    
    @staticmethod   
    def _filter_df_by_similarity(df, similarity_col, min_similarity, min_keep):
        """
        Filters a DataFrame based on similarity scores.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            similarity_col (str): Column name for similarity scores.
            min_similarity (float): Minimum similarity threshold for filtering.
            min_keep (int): Minimum number of rows to keep.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """        
        if len(df) <= min_keep:
            # no filtering needed
            return df

        # get top results
        top_pages = df.sort_values(similarity_col, ascending=False).head(min_keep)
        # select remaining pages by similarity
        remaining_df = df[~df.index.isin(top_pages.index)]
        remaining_df = remaining_df[remaining_df[similarity_col] >= min_similarity]

        # combine top and remaining pages
        return pd.concat([top_pages, remaining_df]).reset_index(drop=True)
    