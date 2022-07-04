from django.shortcuts import render
from django.http.response import HttpResponse
from myapp.models import Profile
import cv2
import torch
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import word2vec
import requests
from bs4 import BeautifulSoup
from selenium import webdriver as wd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

from django.shortcuts import render, redirect

from django.http.response import HttpResponse
import plotly.graph_objects as go 
import plotly.io as po
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from gensim.models import word2vec
plt.rcParams['font.family'] = 'Malgun Gothic'

# Create your views here.
def Main(request):
    return render(request, 'index.html')

# def upload(request):
#     post = Post()
#     post.image = request.FILES['file']
#     post.save()
#  
#     return render(request, 'index.html')

from django.conf import settings
import os

#file_path = os.path.join(settings.FILES_DIR, 'best.pt')
#best_data = open(file_path, "rb").read()

plt.rcParams['font.family'] = 'Malgun Gothic'

def Upload(request):
    form = Profile()
    form.image=request.FILES['image']
    image = form.image
    print(image)
    form.save()
    display = 'show'
    unique_list_kr, tokenized_doc, tok_list = yoloModel(image)
    keyword = Myword2vec(tok_list)
    lda_json = lda_model(keyword, tokenized_doc, image)
    return render(request, 'index.html', {'before_image':image, 'display':display, 'lda_json':lda_json, 'keyword':keyword}) 

def chart_word(request):
    return render(request, '3.html')

def yoloModel(image):
    #best = '../media/images/best.pt'
    #best = '../media/best.pt'
    #best_data = "../../model/best.pt"
    best_data = 'E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=best_data)
    ckpt = torch.load(best_data)['model']  # load checkpoint
    model.names = ckpt.names
    # 'C:/work/psou/final_project/myapp/media/image/party.jpg'
    img1 = Image.open('E:/파이썬 프로젝트 필요한 데이터/final_project/media/'+str(image))  # PIL image
    
    # 'C:/work/psou/final_project/media'+str(image)

    # Inference
    
    results = model(img1, size=640)  # includes NMS
    results.save('E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/images')
    # Results
    results.print()
    #print(results.names)
    word_list = []
    #print('pred')
    #print(results.pred)
    #print('xyxy[0]')
    #print(results.xyxy[0])
    

    # 과학적 표기법 대신 소수점 6자리까지 나타낸다.
    np.set_printoptions(precision=6, suppress=True)
    weight_list = []  
    for i, det in enumerate(results.pred):
        #print(i)
        #print(det)
        # Print results
        for c in det[:, -1].unique():
            #print(int(c))
            word_list.append((results.names[int(c)]))
    
        
        for *xyxy, conf, cls in reversed(det):
            label = '%s %.2f' % (results.names[int(cls)], conf)
            if round(float(conf),2) >= 0.7:
                weight_list.append((results.names[int(cls)]))
    
    
    #print(weight_list)
    unique_set = set(weight_list)
    unique_list = list(unique_set)            
    print('0.7이상 weight 단어:', unique_list)
    
    unique_list_kr = []            
    for word in unique_list :
        if word == 'balloons' : 
            change_kr = word.replace("balloons","풍선")
    
        if word == 'bottle' :  
            change_kr = word.replace("bottle","병")
    
        if word == 'bread' :      
            change_kr = word.replace("bread","빵")
    
        if word == 'bus' :      
            change_kr = word.replace("bus","버스")
    
        if word == 'cake' :  
            change_kr = word.replace("cake","케잌")
    
        if word == 'can' :  
            change_kr = word.replace("can","캔")
    
        if word == 'cookie' :  
            change_kr = word.replace("cookie","쿠키")
    
        if word == 'melon' :  
            change_kr = word.replace("melon","멜론")
    
        if word == 'palm tree' :  
            change_kr = word.replace("palm tree","야자수")
    
        if word == 'person' :  
            change_kr = word.replace("person","사람")
    
        if word == 'pineapple' :  
            change_kr = word.replace("pineapple","파인애플")
    
        if word == 'wine glass' :  
            change_kr = word.replace("wine glass","와인잔")
    
        if word == 'car' :  
            change_kr = word.replace("car","자동차")
                
        unique_list_kr.append(change_kr)
    
    print('0.7이상 weight 단어 한글화:', unique_list_kr)
                
    #print(word_list)
    #print(len(word_list))
    #results.show()  # or .show() .save()
    
    import shutil
    #shutil.rmtree('C:/work/psou/final_project/myapp/static/crawling')
  
    tokenized_doc = []
    for i in range(4):
        tok_list = pd.read_csv('E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/crawling/'+str(i)+'_results.csv')['0'].str.extract(r'([ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+)', expand=False).dropna().tolist()
        tokenized_doc.append(tok_list)
    return unique_list_kr, tokenized_doc, tok_list

def Myword2vec(result):
    import collections
    counts = collections.Counter(result)
    #print(counts)
    #print(counts.most_common(1)[0][0])
    
    keyword = counts.most_common(1)[0][0]  
    #result = []
    #with open('C:/work/psou/final_project2.zip_expanded/final_project2/myapp/static/word2vecimg/'+keyword+'_results.csv', mode='r', encoding='utf_8') as fr:    # csv파일 경로지정
    #    lines = fr.read().split('\n')
    #    imsi2 = (" ".join(lines)).strip()     # 좌우 공백 자르기
    #    result.append(imsi2)
    
    
    
    fileName = '{}.txt'.format(keyword)
    with open(fileName, mode='w', encoding='utf_8') as fw:
        fw.write('\n'.join(result))
    
    
    genObj = word2vec.LineSentence(fileName)
    
    # Word Embedding(단어를 수치화)의 일종으로 word2vec
    model = word2vec.Word2Vec(genObj, size = 100, window=10, min_count=40, sg=1)
    # ** word2vec 옵션 정리 **
    # size : 벡터의 크기를 설정
    # window : 고려할 앞뒤 폭 ( 앞 뒤의 단어를 설정 )
    # min_count : 사용할 단어의 최소빈도 ( 설정된 빈도 이하 단어는 무시 )
    # workers : 동시에 처리할 작업의 수 
    # sg : 0 (CBOW) , 1 (Skip-gram)
    # CBOW 는 주변 단어를 통해 주어진 단어를 예측하는 방법
    # Skip-gram 은 하나의 단어에서 여러 단어를 예측하는 방법
    # 보편적으로 Skip-gram이 많이 쓰인다.
    model.init_sims(replace=True)   # 필요없는 메모리 해제
    
    # 모델 저장
    try:
        model.save('E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/model/'+keyword+'.model')    # 모델저장 경로지정ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ
    except Exception as e:
        print('err : ', e)
        
    # 모델 불러오기
    model = word2vec.Word2Vec.load('E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/model/'+keyword+'.model')    # 모델불러오기 경로지정ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ
    
    # 사진 속의 단어와 가장 유사한 단어 추출 및 DataFrame으로 저장
    result = pd.DataFrame(model.wv.most_similar(keyword, topn = 10), columns=['단어','유사도'])
    
    # 시각화1
    sns.set(style='whitegrid', font='Malgun Gothic', font_scale=1)
    graph = sns.PairGrid(result.sort_values('유사도', ascending=False), y_vars=['단어'])
    graph.fig.set_size_inches(5, 10)
    
    graph.map(sns.stripplot, size = 10, orient = 'h', palette = 'ch:s=1, r=-1, h=1_r', linewidth = 1, edgecolor='w')
    graph.set(xlim = (0.0, 1), xlabel = '유사도', ylabel = '')
    titles = keyword
   
    for ax, title in zip(graph.axes.flat, titles):
        ax.set(title = titles)
        ax.xaxis.grid(False); ax.yaxis.grid(True)    # 수직격자 False, 수평격자 True
    
        sns.despine(left = True, bottom = True)
    plt.savefig('E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/word2vecimg/{}1.png'.format(keyword), bbox_inches='tight') # 경로설정
    #plt.show()
    plt.close()
    
    # 시각화 2
    word_vectors = model.wv
    vocabs = word_vectors.wv.vocab
    vocabs= list(vocabs.keys())
    word_vectors_list = [word_vectors[v] for v in vocabs]
    #print(word_vectors_list)
    
    pca = PCA(n_components = 2)
    xys = pca.fit_transform(word_vectors_list)
    xs = xys[:,0]
    ys = xys[:,1]
    
    #plt.style.use(['dark_background'])
    s = 80
    plt.scatter(xs, ys, marker='o', alpha = 0.7, edgecolors = 'w' , cmap='Green', s = s)
    for i,v in enumerate(vocabs):
        plt.annotate(v,xy=(xs[i], ys[i]))
    plt.grid(False)
    
    plt.savefig('E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/word2vecimg/{}2.png'.format(keyword), bbox_inches='tight') # 경로설정
    #plt.show()
    plt.close()
    
    
    # 시각화3
    # pip install plotly
    fig = go.Figure(data=go.Scatter(x = xs, y = ys, mode='markers+text', text=vocabs, marker=dict(size=80))) 
    fig.update_layout(template='plotly_dark')
    fig.update_layout(title=keyword)
    po.write_html(fig, file ='E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/templates/3.html')  # 경로설정 
    #fig.show()
    return keyword


from django.shortcuts import render
import json
from xgboost.training import cv
from django.conf.locale import id
import os.path                                         
from gensim.test.utils import datapath
import pandas as pd
from gensim import corpora
import gensim

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def lda_model(keyword, tokenized_doc, image):
       
    dictionary = corpora.Dictionary(tokenized_doc)
    #print(dictionary)
    corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
    #print(len(corpus)) # 수행된 결과에서 두번째 출력. 첫번째 문서의 인덱스는 0
    #print([[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]])
    
    def compute_coherence_values(dictionary, corpus, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics
         
        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics
         
        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        perplexity_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=15)
            model_list.append(model)
            coherence_model_lda = gensim.models.CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            coherence_values.append(coherence_model_lda.get_coherence())
            perplexity_values.append(model.log_perplexity(corpus))
        return model_list, coherence_values, perplexity_values
    
    
    # Can take a long time to run.
    model_list, coherence_values, perplexity_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, start=2, limit=8, step=1)
    
    # Show graph
    import matplotlib.pyplot as plt
    limit=8; start=2; step=1;
    x = range(start, limit, step)
    
    
    dic = {}
    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
        dic[abs(round(cv, 4))] = m
    
    for m, pv in zip(x, perplexity_values):
        print("Num Topics =", m, " has perplexity values of", round(pv, 4))
        dic[abs(round(pv, 4))] = m
    
    a = min(dic.keys())
    #print('dic[a]:', dic[a])
       
        
    file = 'E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/model/lda.model'
    #if not os.path.isfile(file):
        #print("Nothing")      
    NUM_TOPICS = dic[a] # dic[a]개의 토픽
    lda = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15, random_state=121)
    lda.save('E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/model/lda.model')
        
    #else :
    fname = datapath("E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/model/lda.model")
    lda = gensim.models.LdaModel.load(fname, mmap='r')
    
    word = gensim.models.coherencemodel.CoherenceModel.top_topics_as_word_lists(lda, dictionary, topn=30)
    print(len(word))
    topics = lda.print_topics(num_words=30)
    for topic in topics:
        print(topic)
    
    dict = {}
    for cp in corpus:
        for id, freq in cp:
            dict[dictionary[id]] = freq                                  
    #print('dict:', dict)
    
    qq = list(dict.keys())
    
    ddd = []
    
    for q in qq:
        for i in range(len(word)):
            for w in word[i]:
                if q == w:
                    
                    ddd.append([i, (dict[q], q)])
                    #print([i, (q, dict[q])])
    #print(ddd)
    
    
    df = pd.DataFrame(columns = list(range(len(word))))
    #print(df)
    
    for i in df.columns: 
        for j in range(len(ddd)):
            if i == ddd[j][0]:
                #print(i, ddd[j][1])
                df.loc[j ,i] = ddd[j][1]
                
    l = []            
    for i in df.columns:
        for j in range(30):
            l.append(j)
    #print(l)
    df['인덱스'] = l
    df = df.set_index('인덱스')
       
    
    se_li = []
    for i in df.columns: 
        se_li.append(df.loc[:, i].dropna().sort_values(ascending=False).reset_index().drop(['인덱스'], axis=1))
    
    re_l = []
    for j in range(30):
        re_l.append(j)
    
    df = pd.concat(se_li, axis=1)
    print(df)
    
    
   
    
    # json 형식으로 변환
    jli = []
    jlis = []
    
    #_, imdata = cv2.imencode('.JPG',red)
    #base64.b64encode(imdata).decode('ascii')
    json_dic = {'name':keyword,
            "children":jli}  
      
    for i in range(len(df.columns)):     
        jli2 = []
        for j in range(5):
            jli2.append({'name': df.iloc[j+1, i][1], 'value': df.iloc[j+1, i][0]})
        jlis.append(jli2)
        jli.append({'name': df.iloc[0, i][1], 'children': jlis[i]})
             
    print(json_dic)
        
    # Compute Perplexity
    print('\nPerplexity: ', lda.log_perplexity(corpus)) # a measure of how good the model is. lower the better.
     
    # Compute Coherence Score
    coherence_model_lda = gensim.models.CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)
    with open('E:/파이썬 프로젝트 필요한 데이터/final_project/myapp/static/json_files/json_dic.json', 'w', encoding="utf-8") as make_file:
        json.dump(json_dic, make_file, ensure_ascii=False, indent="\t")
    # content_type = 'application/json'
    lda_data = json.dumps(json_dic, ensure_ascii=False)
    
    return lda_data  
