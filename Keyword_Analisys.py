from tkinter import *
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from konlpy.tag import Okt
from wordcloud import WordCloud
import os

os.environ['JAVA_HOME'] = os.path.expanduser('C:\Program Files\Java\jdk-20')
search = 0

def news_Search():

    window = Tk()
    window.option_add("*Font", "맑은고딕 10")

    img = PhotoImage(file='heart2.png')
    img_label = Label(window, image=img, bg='pink')
    img_label.grid(column=0, row=0)

    title_label = Label(window, text='뉴스 키워드 검색기', bg="red", font=('나눔바른펜', 18, 'bold'))
    title_label.grid(column=0, row=0)

    entry = Entry(window, width=30)
    entry.grid(column=0, row=1, pady=15)

    def close():
        global search
        search = entry.get()

        window.destroy()

    btn = Button(window, text='검색하기', width=10, bg="pink", command=close)
    btn.grid(column=0, row=2, pady=5)

    window.mainloop()

    return search

# 네이버에 혼인율 검색 후 10 페이지까지 뉴스 제목 스크래핑
def get_Keyword_Analisys():

    list = []
    keyword = news_Search()

    for i in range(1, 100, 10):
        response = requests.get(f"https://search.naver.com/search.naver?where=news&sm=tab_jum&query={keyword}&start={i}")
        soup = BeautifulSoup(response.text,'html.parser')

        links = soup.select('.news_tit')

        for link in links:
            title = link.text
            # url = link.attrs['href'] 뉴스 주소 따오기는 생략
            list.append([title])

    # 스크래핑 한 내용 csv 파일로 저장
    # 처음에는 아래 내용 주석 풀고 Run 하기~
    f = open(f'{keyword}_Newslist.txt','w',encoding='utf-8' ,newline='')
    list = str(list)

    for i in list:
        f.write(i)
        if i ==']':
            f.write('\n')
    f.close()

    #키워드 분석 시작
    with open(f'{keyword}_Newslist.txt', 'r', encoding='utf-8') as d:
        text = d.read()

    okt = Okt()
    output = okt.pos(text) #형태소와 품사 추출

    words = []
    # 형용사와 명사만 리스트에 넣어주기
    for i, j in output:
        if j in['Noun', 'Adjective']:
            words.append(i)

    #제외 할 단어 추가
    stop_words = "또 명 세 의 혼인 혼인율 율 과 더 악 그 률 전 길 별 인 더 안 해도 점 용 시 때 브 새 뚝 수 제 탓 순 간 최 돌 줄 게 사 로 늘 것 무 치 실 나선 낮아 중"
    stop_words = set(stop_words.split(' '))
    #불용어 제외
    words = [i for i in words if not i in stop_words]
    #가장 많이 나온 단어 100개 저장
    count = Counter(words)
    count = count.most_common(100)

     # 키워드 분석 시각화
    mask = Image.new("RGBA",(2500, 2200), (255, 255, 255))
    image = Image.open('heart.png').convert('RGBA')
    x, y = image.size
    mask.paste(image, (0, 0, x, y), image)
    mask = np.array(mask)

    # # 폰트 지정
    font = 'C:/Windows/Fonts/맑은고딕/malgun.ttf'
    word_cloud = WordCloud(font_path=font, background_color='black', max_font_size=400, mask=mask, colormap='prism').generate_from_frequencies(dict(count))
    plt.figure(figsize=(10, 8))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.savefig('혼인율.png',bbox_inches='tight')
    # plt.show()

