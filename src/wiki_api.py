import wikipediaapi
import os

from config import SECTIONS_EXCLUDE

def get_wiki_page_sections_as_dict(page_title, sections_exclude=SECTIONS_EXCLUDE):
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=os.getenv('WIKI_USER_AGENT'), language='en')
    page = wiki_wiki.page(page_title)
    
    if not page.exists():
        return None
    
    def sections_to_dict(sections, parent_titles=[]):
        result = {'Summary': page.summary}
        for section in sections:
            if section.title in sections_exclude: continue
            section_title = ": ".join(parent_titles + [section.title])
            if section.text:
                result[section_title] = section.text
            result.update(sections_to_dict(section.sections, parent_titles + [section.title]))
        return result
    
    return sections_to_dict(page.sections)
