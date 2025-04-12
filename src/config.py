# wikipedia user agent
WIKI_SEARCH_URL = 'https://api.wikimedia.org/core/v1/wikipedia/en/search/page'
# number of pages to return from wikipedia search
N_WIKI_PAGES = 3

# column to select for page similarity calculation
PAGE_SIM_COL = 'excerpt'

# embedding model
EMBEDDING = 'multi-qa-MiniLM-L6-cos-v1'

# name of llm model
OPEN_AI_MODEL = 'gpt-4'
TEMPERATURE = 0.2

# number of keywords for wikipedia search
N_KEYWORDS = 5

# criteria for filtering pages
MIN_PAGE_SIMILARITY = 0.2
MIN_PAGES_KEEP = 3

# criteria for filtering sections
MIN_SECTION_SIMILARITY = 0.2
MIN_SECTIONS_KEEP = 10

# weight of rejection to calculate section score
REJECTION_WEIGHT = 0.5

# sections to be excluded
SECTIONS_EXCLUDE = ['See also', 'References', 'External links', 'Further reading']

# number of sections to sample to create context
N_SECTIONS_SELECT = 1

# number of multiple choice answers
N_CHOICES = 4

# keyword if question generation fails because provided context is not aligned with user query
FAILED_QUERY = 'QUERY FAILED'

# maximum number of atempts to re-generate a question
MAX_RETRIES = 3
