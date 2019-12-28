import sys
import codecs
import re
import json
import math
import code


# 数一数每个中文字出现的频数，并且取对数。
def countChar():
    fi = open('nlpcc-iccpol-2016.kbqa.training-data','r',encoding='utf8')

    q=''

    countChar = {}
    for line in fi:
        if line[1] == 'q': # 计算问题句子
            q = line[line.index('\t') + 1:].strip()
            for char in q:
                if char not in countChar:
                    countChar[char] = 1
                else:
                    countChar[char] += 1
        if line[1] == 't': # 计算sub和pre
            sub = line[line.index('\t') + 1:line.index(' |||')].strip()
            qNSub = line[line.index(' ||| ') + 5:]
            pre = qNSub[:qNSub.index(' |||')]
            for char in sub+pre:
                if char not in countChar:
                    countChar[char] = -1
                else:
                    countChar[char] = countChar[char] - 1


    fo = open('countChar','w',encoding='utf8')

    for key in list(countChar):
        if countChar[key] < 1:
            del countChar[key]
        else:
            countChar[key] = math.log10(countChar[key])

    json.dump(countChar,fo)

    fo.close()


    # 可以在这个文件中看到每个字的log频数
    fotxt = open('countChar.txt','w',encoding='utf8')

    for pair in sorted(countChar.items(), key=lambda d:d[1],reverse=True):
        fotxt.write(pair[0] + ':'+str(pair[1]) + '\n')


    fotxt.close()


countChar()

def loadKB(path, encode = 'utf8'):
        
    fi = open(path, 'r', encoding=encode)
    prePattern = re.compile(r'[·•\-\s]|(\[[0-9]*\])')


    kbDict={}
    newEntityDic={}
    i = 0
    for line in fi:
        # 河北外国语职业学院 ||| 别名 ||| 河北外国语职业学院
        i += 1
        print('exporting the ' + str(i) + ' triple', end='\r', flush=True)
        entityStr = line[:line.index(' |||')].strip() # subject ||| predicate ||| object
        # 上面这一行得到了 subject

        tmp = line[line.index('||| ') + 4:]
        relationStr = tmp[:tmp.index(' |||')].strip() # predicate
        relationStr, num = prePattern.subn('', relationStr)
        objectStr = tmp[tmp.index('||| ') + 4:].strip() # object
        if relationStr == objectStr: #delete the triple if the predicate is the same as object
            continue
        if entityStr not in kbDict:
            newEntityDic = {relationStr:objectStr}
            kbDict[entityStr] = []
            kbDict[entityStr].append(newEntityDic)
        else:
            kbDict[entityStr][len(kbDict[entityStr]) - 1][relationStr] = objectStr
            # 其实我不确定这里为什么要使用这个奇怪的数据结构
            

    fi.close()

    
    return kbDict


print("load KB...")
kbDictRaw = loadKB('nlpcc-iccpol-2016.kbqa.kb')
# code.interact(local=locals())

def addAliasForKB(kbDictRaw):
    # 在kb中增加alias
    pattern = re.compile(r'[·•\-\s]|(\[[0-9]*\])')

    patternSub = re.compile(r'(\s*\(.*\)\s*)|(\s*（.*）\s*)')  # subject需按照 subject (Description) || Predicate || Object 的方式抽取, 其中(Description)可选

    patternBlank = re.compile(r'\s')

    patternUpper = re.compile(r'[A-Z]')

    patternMark = re.compile(r'《(.*)》')

    kbDict = kbDictRaw.copy()
    for key in list(kbDict):
        if patternSub.search(key):
            keyRe, num = patternSub.subn('', key)
            # 把括号内的东西去掉
            if keyRe not in kbDict:
                kbDict[keyRe] = kbDict[key]
            else:
                for kb in kbDict[key]:
                    kbDict[keyRe].append(kb)


    for key in list(kbDict):
        match = patternMark.search(key)
        # 把书名号括号里的东西去掉
        if match:
            keyRe, num = patternMark.subn(r'\1', key)
            if keyRe not in kbDict:
                kbDict[keyRe] = kbDict[key]
            else:
                for kb in kbDict[key]:
                    kbDict[keyRe].append(kb)


    for key in list(kbDict):
        if patternUpper.search(key):
            keyLower = key.lower()
            # 添加小写
            if keyLower not in kbDict:
                kbDict[keyLower] = kbDict[key]
            else:
                for kb in kbDict[key]:
                    kbDict[keyLower].append(kb)

    for key in list(kbDict):
        if patternBlank.search(key):
            # /s去掉
            keyRe, num = patternBlank.subn('', key)
            if keyRe not in kbDict:
                kbDict[keyRe] = kbDict[key]
            else:
                for kb in kbDict[key]:
                    kbDict[keyRe].append(kb)
    
    return kbDict   


print('Cleaning kb......')
kbDict = addAliasForKB(kbDictRaw)
json.dump(kbDict, open('kbJson.cleanPre.alias.utf8','w',encoding='utf8'))
print('\nDone!')
# code.interact(local=locals())



#把文本格式的word vector导出成Json格式供后续读入为Python的Dictionary
def convertToJson(inputPath='vec_zhwiki_300mc20.txt', outputPath='vectorJson.utf8'\
                  ,encode = 'utf8'):
    fi = open(inputPath,'r',encoding=encode)

    ll = []
    for line in fi:
        ll.append(line.strip())
    listTmp = []

    embeddingDict = {}
    for i in range(len(ll)-1):
        lineTmp = ll[i+1]
        listTmp = []
        indexSpace = lineTmp.find(' ')
        embeddingDict[lineTmp[:indexSpace]] = listTmp
        lineTmp = lineTmp[indexSpace + 1:]
        for j in range(300):
            indexSpace = lineTmp.find(' ')
            listTmp.append(float(lineTmp[:indexSpace]))
            lineTmp = lineTmp[indexSpace + 1:]



    print('Vector size is ' + str(len(listTmp)))
    print('Dictionary size is ' + str(len(embeddingDict)))
            
    json.dump(embeddingDict,open(outputPath,'w',encoding=encode))

print('Dumping word vector to Json format......')
convertToJson()
print('Done!')



#用训练数据训练答案模板
def getAnswerPatten(inputPath = 'nlpcc-iccpol-2016.kbqa.training-data', outputPath = 'outputAP'):
    inputEncoding = 'utf8'
    outputEncoding = 'utf8'

    fi = open(inputPath, 'r', encoding=inputEncoding)
    fo = open(outputPath, 'w', encoding=outputEncoding)

    qRaw = ''


    pattern = re.compile(r'[·•\-\s]|(\[[0-9]*\])') #pattern to clean predicate, in order to be consistent with KB clean method

    APList = {}
    APListCore = {}
    for line in fi:
        if line.find('<q') == 0:  #question line
            qRaw = line[line.index('>') + 2:].strip()
            continue
        elif line.find('<t') == 0:  #triple line
            triple = line[line.index('>') + 2:]
            s = triple[:triple.index(' |||')].strip()
            triNS = triple[triple.index(' |||') + 5:]
            p = triNS[:triNS.index(' |||')]
            p, num = pattern.subn('', p)
            if qRaw.find(s) != -1:
                qRaw = qRaw.replace(s,'(SUB)', 1)
            # 把问题中出现的subject变成(SUB)

            # <question id=1>   《机械设计基础》这本书的作者是谁？
            # <triple id=1>   机械设计基础 ||| 作者 ||| 杨可桢，程光蕴，李仲生
            # <answer id=1>   杨可桢，程光蕴，李仲生

            #  《(SUB)》这本书的作者是谁？ ||| 作者
           
            qRaw = qRaw.strip() +  ' ||| '  + p
            if qRaw in APList:
                APList[qRaw] += 1
            else:
                APList[qRaw] = 1

     
        else: continue

    json.dump(APList, fo)

    fotxt = open(outputPath+'.txt', 'w', encoding=outputEncoding)

    for key in APList:
        fotxt.write(key + ' ' + str(APList[key]) + '\n')
        
    fotxt.close()

    fi.close()    
    fo.close()

print('Training answer pattern......')
getAnswerPatten()
print('Done!')


