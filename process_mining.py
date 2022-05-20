import pandas as pd
import re
import pm4py
import graphviz


def filter_affected_app_logs(application, logs):
    '''
        filter_affected_app_logs(application, logs)
        filters out the logs for the specified affected application
        >>> parameters:
            - application | string | name of affected application
        >>> returns:
            - app_data | dataframe | ...of incident logs of this app
    '''
    app_data = logs[logs['Impacted application']==application][['Short Description','Work notes']].reset_index(drop=True) #later add incident sub_type
    return app_data   

def preprocess_worknotes(log):
    '''
        preprocess_worknotes(log)
        converts the input log into a dataframe of time and worknotes.
        >>> parameters:
            - logs | string | name of affected application
        >>> returns: 2D list of [incident_Number, timestamp, workNote]'s
    '''
    incident_ID = log['Number']
    arr = []
    for string in re.split(r'\s+(?=\d+-\d{2})', log['Work notes']):
        temp_lis = re.split('\n',string)
        timestamp = re.findall('^(.+)-',temp_lis.pop(0))[0]
        worknote = clean_text(' '.join(temp_lis))
        arr.append([incident_ID,timestamp,worknote])
    return arr

def clean_text(s):
    '''
        takes a string and returns it after cleaning.
    '''
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: "&gt", "&lt"
    s = re.sub(r'&gt|&lt', ' ', s)
    # normalization 4: letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # normalization 5: non-word repetition (if more than 1)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    #s = re.sub(r'\(.*?\)', '. ', s)
    # normalization 8: xxx[?!]. -- > xxx.
    s = re.sub(r'\W+?\.', '.', s)
    # normalization 9: [.?!] --> [.?!] xxx
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    # normalization 12: phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)
    # removing all punctuation other than '.\n'
    s = re.sub(r'[^\w\n\d :\\.-]','',s)

    return s.strip()

def draw_process_flow(process_data):

    data = pm4py.format_dataframe(process_data, case_id='Number', activity_key='workNote', timestamp_key=\
        'Timestamp')

    dfg, starts, ends = pm4py.discover_dfg(data)

    node_list = []
    for edge in dfg.keys():
        node_list.extend(list(edge))
    node_list = list(set(node_list))

    node_dict = {}
    for node in node_list:
        if node not in node_dict:
            node_dict[node] = 0
        else:
            pass

    for edge, freq in dfg.items():
        node_dict[edge[0]]+=freq
        node_dict[edge[1]]-=freq 

    node_dict = {k: v for k, v in sorted(node_dict.items(), reverse=True, key=lambda item: item[1])}

    graph = graphviz.Digraph()

    i = 0
    for nodes in node_dict.keys():
        graph.node(nodes, shape = 'oval')#, pos = ("0,"+str(-1*i)))
        i+=3

    for edge, freq in dfg.items():
        graph.edge(tail_name=edge[0], head_name= edge[1])

    graph.render('process_graph', format='png', view=True, engine='fdp')

    return starts,ends

def classify_worknote(str_):
    '''
        classify worknote based on start words | these are identified as standeard worknote types in IT
    '''
    types = ['Assignment','Automatic Assignment','Remarks','Request Info','Ask if','Response was','Take Action','Action outcome',\
            'Escalate to','User reported','Raise','Resolved','Resolution Note']
    types = [x.lower() for x in types]

    for type_ in types:
        if str_.startswith(type_):
            return type_
    return 'unformatted'