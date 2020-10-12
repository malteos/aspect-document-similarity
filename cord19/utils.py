# normalize section title
def normalize_section(title):
    return title.strip().lower()\
        .replace('conclusions', 'conclusion')\
        .replace('concluding remarks', 'conclusion')\
        .replace('future perspectives', 'future work')\
        .replace('future directions', 'future work')\
        .replace('viruses.', 'virus')\
        .replace('viruses', 'virus')
        #.replace('conclusion and future perspectives', 'conclusion')\
        #.replace('materials and methods', 'methods')


def resolve_and_sect_titles(cits):
    for from_doi, to_doi, sect_title in cits:
        for t in normalize_section(sect_title).split(' and '):
            yield (from_doi, to_doi, t)


def get_text_from_doi(doi, doi2paper, raise_not_found_error=True):
    text = ''
    sep = '\n'

    # if doi in doi2s2paper:
    #     # from s2 scraper
    #     # text += doi2s2paper[doi]['title']
    #
    #     if doi2s2paper[doi]['abstract']:
    #         # text += '\n' + doi2s2paper[doi]['abstract']
    #         text = doi2s2paper[doi]['title'] + sep + doi2s2paper[doi]['abstract']

    if doi in doi2paper:
        # text += doi2paper[doi]['metadata']['title']

        if doi2paper[doi]['abstract'] and len(doi2paper[doi]['abstract']) > 10:
            # text += doi2paper[doi]['metadata']['title'] + '\n' + doi2paper[doi]['abstract'][0]['text']
            text = doi2paper[doi]['title'] + sep + doi2paper[doi]['abstract']

    elif raise_not_found_error:
        raise ValueError('DOI not found')

    return text
