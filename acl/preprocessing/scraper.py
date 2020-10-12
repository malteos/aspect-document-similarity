import requests
import tqdm


def scrape_s2(job_name, needed_ids, id2s2, id2s2_errors, id_prefix='', sleep=2.5, save_every_n=1000, offset=0):
    api_url = 'http://api.semanticscholar.org/v1/paper/'

    try:
        for i, needed_id in enumerate(tqdm(needed_ids, total=len(needed_ids))):
            if i < offset:  # skip
                continue

            if needed_id in id2s2 or needed_id in id2s2_errors:
                continue

            res = requests.get(api_url + id_prefix + needed_id)

            if res.status_code == 200:
                try:
                    id2s2[needed_id] = res.json()
                except ValueError as e:
                    print(f'Error cannot parse JSON: {needed_id}')
                    id2s2_errors[needed_id] = str(e)
            elif res.status_code == 429:
                print(f'Stop! Rate limit reached at: {i}')
                break
            elif res.status_code == 403:
                print(f'Stop! Forbidden / rate limit reached at: {i}')
                break
            elif res.status_code == 404:
                id2s2_errors[needed_id] = 404
            else:
                print(f'Error status: {res.status_code} - {needed_id}')
                id2s2_errors[needed_id] = res.text

            if save_every_n > 0 and (i % save_every_n) == 0 and i > 0:
                json.dump(id2s2, open(output_dir / f'{job_name}.json', 'w'))
                json.dump(id2s2_errors, open(output_dir / f'{job_name}_errors.json', 'w'))

            time.sleep(sleep)
    except KeyboardInterrupt:
        print('Aborting...')
        pass

    return id2s2, id2s2_errors



def scrape_dblp():
    missing_titles = set(filtered_cits.keys()).difference(set(title2dblp_hits.keys()))
    print(f'Missing titles: {len(missing_titles):,}')

    title2dblp_hits = {}
    dblp_errors = {}

    url = 'https://dblp.org/search/publ/api'

    for i, (title, idxs) in tqdm(enumerate(filtered_cits.items()), total=len(filtered_cits)):
        if title in title2dblp_hits or title in dblp_errors:
            continue

        q = title
        res = requests.get(url, params={'query': q, 'format': 'json'})

        if res.status_code == 200:
            title2dblp_hits[title] = res.json()['result']['hits']
        elif res.status_code == 422:
            dblp_errors[title] = res.status_code
            print(f'422: unprocesseble entity: {title}')
        else:
            # dblp_errors[title] = res.status_code
            print(f'Error: {res.text}')
            break

        time.sleep(0.5)

        # if i > 3:
        #    break

    print(f'Scraped data for {len(title2dblp_hits)} papers from DBPL (errors: {len(dblp_errors)})')
