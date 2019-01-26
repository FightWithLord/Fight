import sys
import jieba.analyse
import jieba.posseg as pseg

def main(argv):
    #print("argv", argv)
    content = u'中国特色社会主义是我们党领导的伟大事业'
    content = argv[1]
    print("argv",content)
    words = pseg.cut(content)
    for word, flag in words:
        print('%s, %s' % (word, flag))
    keywords = jieba.analyse.textrank(content, topK=20, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
    for item in keywords:
        # 分别为关键词和相应的权重
        print (item[0], item[1])


if __name__ == '__main__':
    main(sys.argv)
