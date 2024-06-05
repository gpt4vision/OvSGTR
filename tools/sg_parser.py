import os
import sys
import json
import sng_parser
import spacy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import re

try:
    parser = sng_parser.Parser('spacy', model='en')
    print("use spacy parser")
except:
    import sng_parser as parser
    print("import sng_parser as parser !")

black_lists = set(['we', 'me',  'i', 'you', 'u', 'he', 'she', 'them', 'her', 'his',
                    'they', 'this', 'that', 'it', 'image', 'group',
                    'view', 'close', 'middle', 'side', 'center', 'color',
                    'top', 'which', 'where', 'who', 'some', 'someone',
                    'front', 'back', 'type', 'air', 'day', 'time'])


DEBUG = True #if os.environ.get("DEBUG") == '1' else False 

CAPTION_FILE = "./data/coco/annotations/captions_train2017.json"
DST_FILE = "./data/coco/annotations/captions_train2017_triple.json"

#CAPTION_FILE = sys.argv[1]
#DST_FILE = sys.argv[2]


def correct(string):
    string = string.replace("t.v.", 'tv').replace("st.", "street")
    res = all(x.isalpha() or x.isspace() for x in string) # only a-z, A-Z, and whitespace characters are valid
    flag = res 
    if not flag:
        print("invalid:", string)

    return flag



def get_triple(data):
    if 'regions' in data:
        captions = [e['phrase'] for e in data['regions']]
    else:
        captions = data['caption']

    is_single = False
    if isinstance(captions, str):
        captions = [captions]
        is_single = True

    all_rels = []
    for caption in captions:
        graph = parser.parse(caption)

        entities_1 = [item['lemma_head'] for item in graph['entities']]
        #entities = [item['span'] for item in graph['entities']]
        if len(entities_1) == 0:
            import pdb; pdb.set_trace()

        rels_1 = [(entities_1[item['subject']].lower(), 
                 entities_1[item['object']].lower(), 
                 item['lemma_relation'].lower() ) for item in graph['relations']]

        #rels = [(entities[item['subject']].lower(), 
        #         entities[item['object']].lower(), 
        #         item['relation'].lower() ) for item in graph['relations']]
        rels = rels_1
        rels = [e for e in rels if e[0] not in black_lists and correct(e[0]) 
                        and e[1] not in black_lists and correct(e[1])
                        and correct(e[2])]

        if len(rels) > 0:
            rels = list(set(rels))
            all_rels.append(rels)


    try:
        data['relations'] = all_rels[0] if is_single else all_rels
    except:
        data['relations'] = []

    if DEBUG:
        #import pdb; pdb.set_trace()
        pass

    if 'regions' in data:
        return {'image_id': data['id'], 'relations': data['relations']}

    return data

def read_json(name):
    is_jsonl = name.endswith(".jsonl")
    if is_jsonl:
        with open(name, 'r') as fin:
            lines = fin.readlines()
        data = [json.loads(line) for line in lines]
    else:
        with open(name, 'r') as fin:
            data = json.load(fin)

    return data

def process_data(data, num=20):
    if DEBUG:
        return [get_triple(e) for e in data]

    pool = Pool(num)
    data = pool.map(get_triple, data)
    pool.close()
    pool.join()
    return data

def main():
    # Read caption file
    data = read_json(CAPTION_FILE)
    if 'annotations' in data:
        anns = data['annotations']
    else:
        anns = data

    print("Total data:", len(anns))
    processed_data = process_data(tqdm(anns), 30)
    is_jsonl = CAPTION_FILE.endswith(".jsonl")

    processed_data = [e for e in tqdm(processed_data) if len(e['relations']) > 0]
    print("Total processed data:", len(processed_data))
    print(processed_data[0])
    
    is_single = False
    try:
        is_single = True if isinstance(processed_data[0]['caption'], str)  else False
    except:
        pass

    nouns, relations = {}, {}
    for ann in processed_data:
        rels = ann['relations']
        if not is_single:
            tmp = []
            for e in rels:
                tmp.extend(e)
            rels = tmp

        for rel in rels:
            if rel[0] not in nouns:
                nouns[rel[0]] = 0
            if rel[1] not in nouns:
                nouns[rel[1]] = 0

            nouns[rel[0]] += 1
            nouns[rel[1]] += 1

            if len(rel) != 3:
                print("*"*10, rel)

            if rel[2] not in relations:
                relations[rel[2]] = 0
            relations[rel[2]] += 1

    nouns = sorted(nouns.items(), key=lambda x: x[1], reverse=True)
    relations = sorted(relations.items(), key=lambda x: x[1], reverse=True)

    print("Total nouns:", len(nouns), " top 5:", nouns[:5])
    print("Total relations:", len(relations), " top 5:", relations[:5])

    # remove empty
    processed_data = [e for e in tqdm(processed_data) if len(e['relations']) > 0]

    dst = [e[0] + ',' + str(e[1]) + '\n' for e in nouns]
    with open("parsed_nouns.txt", 'w') as fout:
        fout.writelines(dst)

    dst = [e[0] + ',' + str(e[1]) + '\n' for e in relations]
    with open("parsed_relations.txt", 'w') as fout:
        fout.writelines(dst)

    if is_jsonl:
        dst = [json.dumps(line) + '\n' for line in processed_data]
        with open(DST_FILE, 'w') as fout:
            fout.writelines(dst)
    else:
        with open(DST_FILE, 'w') as fout:
            json.dump(processed_data, fout)

if __name__ == "__main__":
    main()

