import sys
import codecs
# import lcs   #the lcs module is in extension folder
import time
import json
from scipy.spatial.distance import cosine
import code


class answerCandidate:
    def __init__(self, sub = '', pre = '', qRaw = '', qType = 0, score = 0, kbDict = [], wS = 1, wP = 10, wAP = 100):
        self.sub = sub # subject 
        self.pre = pre # predicate
        self.qRaw = qRaw # raw question
        self.qType = qType # question type
        self.score = score # 分数
        self.kbDict = kbDict # kd dictionary
        self.origin = '' 
        self.scoreDetail = [0,0,0,0,0]
        self.wS = wS # subject的权重
        self.wP = wP # predicate的权重
        self.wAP = wAP # answer pattern的权重
        self.scoreSub = 0
        self.scoreAP = 0
        self.scorePre = 0
        
    def calcScore(self, qtList, countCharDict, debug=False, includingObj = [], vectorDict = {}):
        # calcScore(问题，candidate_triple) --> score
        # 基于character overlap, term frequency, word2vec word similarity的一套打分方法。
        # 思考：能否设计更好的打分函数，来对我们的candidate triple做ranking呢？
        # calcScore(问题，candidate_triple) --> score
        # 可以考虑的模型:
        # - BERT
        # - EMLo
        # 训练数据的构造方法：针对每一个问题，训练数据提供了正确的triple,我们需要自己来构造一些错误的triple
        # - binary cross entropy loss
        # - scoring问题: hinge loss, max margin loss


        # 最重要的部分，计算该答案的分数
        # qtList表示answer pattern: {question(SUB)|||predicate: count}
        lenSub = len(self.sub)
        scorePre = 0
        scoreAP = 0
        pre = self.pre
        q = self.qRaw
        subIndex = q.index(self.sub)
        qWithoutSub1 = q[:subIndex] # subject左边的部分
        qWithoutSub2 = q[subIndex+lenSub:] # subject右边的部分

        qWithoutSub = q.replace(self.sub,'') # 去掉subject剩下的部分
        qtKey = (self.qRaw.replace(self.sub,'(SUB)',1) + ' ||| ' + pre) # 把subject换成(sub)然后加上predicate
        if qtKey in qtList:
            scoreAP = qtList[qtKey] # 查看当前的问题有没有在知识库中出现过
                    
        self.scoreAP = scoreAP

        # 首先把qWithoutSub都转换成sets，拿出里面所有的不重复的词。
        qWithoutSubSet1 = set(qWithoutSub1)
        qWithoutSubSet2 = set(qWithoutSub2)
        qWithoutSubSet = set(qWithoutSub)
        preLowerSet = set(pre.lower()) # 小写predicate


        # 找出predicate和问题前后两部分的最大intersection
        # 想知道我们的问题和当前的predicate究竟有多高的重合度
        intersection1 = qWithoutSubSet1 & preLowerSet
        intersection2 = qWithoutSubSet2 & preLowerSet

        if len(intersection1) > len(intersection2):
            maxIntersection = intersection1
        else:
            maxIntersection = intersection2

        # 计算来自predicate的分数，采用最大overlap的character的倒数 1/(n+1)
        preFactor = 0
        for char in maxIntersection:
            if char in countCharDict: # 每个字在问题中出现的次数减去它在subject和predicate中出现的次数
                preFactor += 1/(countCharDict[char] + 1) # 比较罕见的字会得到比较高的分数
            else:
                preFactor += 1

        if len(pre) != 0:
            scorePre = preFactor / len(qWithoutSubSet | preLowerSet)
        else:
            scorePre = 0

        
        # 如果上述方法给我们的分数是0的话，就继续使用所有可能的objects来打分，并且按照分数最高的object的分数为准。
        if len(includingObj) != 0 and scorePre == 0:

            for objStr in includingObj: # 所有可能的候选答案
                # 下面这一大段逻辑和predicate计算分数是完全相同的，只是把predicate换成了object

                scorePreTmp = 0
                preLowerSet = set(objStr.lower())
                intersection1 = qWithoutSubSet1 & preLowerSet
                intersection2 = qWithoutSubSet2 & preLowerSet

                if len(intersection1) > len(intersection2):
                    maxIntersection = intersection1
                else:
                    maxIntersection = intersection2

                preFactor = 0
                for char in maxIntersection:
                    if char in countCharDict:
                        preFactor += 1/(countCharDict[char] + 1)
                    else:
                        preFactor += 1

                scorePreTmp = preFactor / len(qWithoutSubSet | preLowerSet)
                if scorePreTmp > scorePre:
                    scorePre = scorePreTmp


        # <question id=1> 你知道计算机应用基础这本书的作者是谁吗？
        # --> {你，知，道，知道，计，算，计算，计算机，机....}
        # <triple id=1>   计算机应用基础 ||| 作者 ||| 秦婉，王蓉
        # 作者 --> {作, 者, 作者}
        if len(vectorDict) != 0 and len(pre) != 0:
            scorePre = 0

            # 找出所有在predicate中出现过的单词的词向量
            segListPre = []
            lenPre = len(pre)
            lenPreSum = 0
            for i in range(lenPre):
                for j in range(lenPre):
                    if i+j < lenPre:
                        preWordTmp = pre[i:i+j+1]
                        if preWordTmp in vectorDict:
                            segListPre.append(preWordTmp)
                            lenPreSum += len(preWordTmp)
                
            # 找出所有在question当中出现过的单词的词向量 
            lenQNS = len(qWithoutSub)
            segListQNS = []
            for i in range(lenQNS):
                for j in range(lenQNS):
                    if i+j < lenQNS:
                        QNSWordTmp = qWithoutSub[i:i+j+1]
                        if QNSWordTmp in vectorDict:
                            segListQNS.append(QNSWordTmp)

            # Add Question type rules, ref to Table.1 in the article                
            if qWithoutSub.find('什么时候') != -1 or qWithoutSub.find('何时') != -1:
                segListQNS.append('日期')
                segListQNS.append('时间')			
            if qWithoutSub.find('在哪') != -1:
                segListQNS.append('地点')
                segListQNS.append('位置')			
            if qWithoutSub.find('多少钱') != -1:
                segListQNS.append('价格')

            # def cosine_similarity(v1, v2):
            #     return np.sum(v1*v2)/np.sqrt(np.sum(v1*v1)) / np.sqrt(np.sum(v2*v2))

            # 计算predicate和question之间的词向量cosine similarity 
            for preWord in segListPre: # {作, 者, 作者} lenPreSum = 4
                scoreMaxCosine = 0
                for QNSWord in segListQNS: # {你，知，道，知道，计，算，计算，计算机，机....}
                    # cosineTmp = lcs.cosine(vectorDict[preWord],vectorDict[QNSWord])
                    # cosineTmp = 1 - scipy.spatial.distance.cosine(vectorDict[preWord],vectorDict[QNSWord])
                    cosineTmp = 1 - cosine(vectorDict[preWord],vectorDict[QNSWord])
                    if cosineTmp > scoreMaxCosine:
                        scoreMaxCosine = cosineTmp
                scorePre += scoreMaxCosine * len(preWord)

            if lenPreSum == 0:
                scorePre = 0
            else:
                scorePre = scorePre / lenPreSum

            self.scorePre = scorePre            

        scoreSub = 0 

        # 计算subject的权重有多高，可能有些subject本身就是更重要一些，一般来说越罕见的entity重要性越高
        for char in self.sub:
            if char in countCharDict:
                scoreSub += 1/(countCharDict[char] + 1) # 
            else:
                scoreSub += 1

        self.scoreSub = scoreSub # 罕见的字在subject中出现会得到比较高的分数
        self.scorePre = scorePre

        # entity的长度，predicate的分数，answer pattern出现的次数
        self.score = scoreSub * self.wS + scorePre * self.wP + scoreAP * self.wAP
        
        return self.score

def getAnswer(sub, pre, kbDict):
    answerList = []
    # kbDict[entityStr][len(kbDict[entityStr]) - 1][relationStr] = objectStr
    # 每个subject都有一系列的KB tiples，然后我们找出所有的subject, predicate, object triples
    for kb in kbDict[sub]:
        if pre in kb:
            answerList.append(kb[pre])
   
    return answerList

def answerQ (qRaw, lKey, kbDict, qtList, countCharDict, vectorDict, wP=10, threshold=0, debug=False):
    q = qRaw.strip().lower() # 问题转化成小写
    # q: 你知道计算机应用基础这本书的作者是谁吗？
    candidateSet = set()
    result = ''
    maxScore = 0
    bestAnswer = set()

    # Get all the candidate triple
    # kbDict[entityStr][len(kbDict[entityStr]) - 1][relationStr] = objectStr
    # 找出所有可能的subject。
    for key in lKey: # 逐个搜索我们的entityStr，也就是knowledge base中所有出现过的subjects
        if -1 != q.find(key): # 如果问题中出现了该subject，那么我们就要考虑这个subject的triples
            for kb in kbDict[key]:
                for pre in list(kb):
                    newAnswerCandidate = answerCandidate(key, pre, q, wP=wP) # 构建一个新的answer candidate
                    candidateSet.add(newAnswerCandidate)
   
    # 以上代码做的事情就是找到所有在question当中出现过的subject, 然后把它们所有的triples都加入到候选答案中去。   
    
    
    candidateSetCopy = candidateSet.copy()
    if debug:
        print('len(candidateSet) = ' + str(len(candidateSetCopy)), end = '\r', flush=True)
    candidateSet = set()

    candidateSetIndex = set()

    for aCandidate in candidateSetCopy:
        strTmp = str(aCandidate.sub+'|'+aCandidate.pre) 
        # 计算机应用基础|作者
        if strTmp not in candidateSetIndex:
            candidateSetIndex.add(strTmp)
            candidateSet.add(aCandidate)

    # 针对每一个问题，以及它对应的<subject, prediction> pair，计算一个分数。

    # 针对每一个candidate answer，计算该candidate的分数，然后选择分数最高的作为答案
    for aCandidate in candidateSet:
        scoreTmp = aCandidate.calcScore(qtList, countCharDict,debug)
        if scoreTmp > maxScore:
            maxScore = scoreTmp
            bestAnswer = set()
        if scoreTmp == maxScore:
            bestAnswer.add(aCandidate)
    
    # 去除一些重复的答案        
    bestAnswerCopy = bestAnswer.copy()
    bestAnswer = set()
    for aCandidate in bestAnswerCopy:
        aCfound = 0
        for aC in bestAnswer:
            if aC.pre == aCandidate.pre and aC.sub == aCandidate.sub:
                aCfound = 1
                break
        if aCfound == 0:
            bestAnswer.add(aCandidate)

    # 加入object的分数
    bestAnswerCopy = bestAnswer.copy()
    for aCandidate in bestAnswerCopy:
        if aCandidate.score == aCandidate.scoreSub:
            scoreReCal = aCandidate.calcScore(qtList, countCharDict,debug, includingObj=getAnswer(aCandidate.sub, aCandidate.pre, kbDict))
            if scoreReCal > maxScore:
                bestAnswer = set()
                maxScore = scoreReCal
            if scoreReCal == maxScore:
                bestAnswer.add(aCandidate)

    # 加入cosine similarity
    bestAnswerCopy = bestAnswer.copy()
    if len(bestAnswer) > 1: # use word vector to remove duplicated answer
        for aCandidate in bestAnswerCopy:
            scoreReCal = aCandidate.calcScore(qtList, countCharDict,debug, includingObj=getAnswer(aCandidate.sub, aCandidate.pre, kbDict), vectorDict=vectorDict)
            if scoreReCal > maxScore:
                bestAnswer = set()
                maxScore = scoreReCal
            if scoreReCal == maxScore:
                bestAnswer.add(aCandidate)
            
    if debug:
        for ai in bestAnswer:
            for kb in kbDict[ai.sub]:
                if ai.pre in kb:
                    print(ai.sub + ' ' +ai.pre + ' '+ kb[ai.pre])
        return[bestAnswer,candidateSet]       
    else:
        return bestAnswer
        


def loadQtList(path, encode = 'utf8'):
    qtList = json.load(open(path,'r',encoding=encode))

    return qtList

def loadcountCharDict(path, encode = 'utf8'):
    countCharDict = json.load(open(path,'r',encoding=encode))

    return countCharDict

def loadvectorDict(path, encode = 'utf8'):
    vectorDict = json.load(open(path,'r',encoding=encode))

    return vectorDict  

def answerAllQ(pathInput, pathOutput, lKey, kbDict, qtList, countCharDict, vectorDict, qIDstart=1, wP=10):
    # lKey: list of all subjects, keys to kbDict
    fq = open(pathInput, 'r', encoding='utf8')
    i = qIDstart
    timeStart = time.time()
    fo = open(pathOutput, 'w', encoding='utf8')
    fo.close()
    listQ = []
    for line in fq:
        if line[1] == 'q':
            listQ.append(line[line.index('\t')+1:].strip())
    for q in listQ:
        fo = open(pathOutput, 'a', encoding='utf8')
        result = answerQ(q, lKey, kbDict, qtList, countCharDict, vectorDict, wP=wP)
        fo.write('<question id='+str(i)+'>\t' + q.lower() + '\n')
        answerLast = ''
        if len(result) != 0:
            answerSet = []
            fo.write('<triple id='+str(i)+'>\t')
            for res in result:
                answerTmp = getAnswer(res.sub, res.pre, kbDict)
                answerSet.append(answerTmp)
                fo.write(res.sub.lower() + ' ||| ' + res.pre.lower() + ' ||| '\
                         + str(answerTmp)  + ' ||| ' + str(res.score) + ' ====== ')
            fo.write('\n')
            fo.write('<answer id='+str(i)+'>\t')

            answerLast = answerSet[0][0]
            mulAnswer = False
            for ansTmp in answerSet:
                for ans in ansTmp:
                    if ans != answerLast:
                        mulAnswer = True
                        continue
                if mulAnswer == True:
                    continue

            if mulAnswer == True:
                for ansTmp in answerSet:
                    for ans in ansTmp:
                        fo.write(ans)
                        if len(ansTmp) > 1:
                            fo.write(' | ')
                    if len(answerSet) > 1:
                        fo.write(' ||| ')
            else:
                fo.write(answerLast)
                
            fo.write('\n==================================================\n')
        else:
            fo.write('<triple id='+str(i)+'>\t')
            fo.write('\n')
            fo.write('<answer id='+str(i)+'>\t')
            fo.write('\n==================================================\n')
        print('processing ' + str(i) + 'th Q.\tAv time cost: ' + str((time.time()-timeStart) / i)[:6] + ' sec', end = '\r', flush=True)
        fo.close()
        i += 1
    fq.close()       
    

def loadResAndanswerAllQ(pathInput, pathOutput, pathDict, pathQt, pathCD, pathVD, encode='utf8', qIDstart=1, wP=10):
    print('Start to load kbDict from json format file: ' + pathDict)
    kbDict = json.load(open(pathDict, 'r', encoding=encode)) # kbJson.cleanPre.alias.utf8
    print('Loaded kbDict completely! kbDic length is '+ str(len(kbDict)))
    qtList = loadQtList(pathQt, encode) # outputAP
    print('Loaded qtList completely! qtList length is '+ str(len(qtList)))
    countCharDict = loadcountCharDict(pathCD) # countChar
    print('Loaded countCharDict completely! countCharDict length is '+ str(len(countCharDict)))
    vectorDict = loadvectorDict(pathVD) # vectorJson.utf8
    # code.interact(local=locals())
    print('Loaded vectorDict completely! vectorDict length is '+ str(len(vectorDict)))
    answerAllQ(pathInput, pathOutput, list(kbDict), kbDict, qtList, countCharDict, vectorDict, qIDstart=1,wP=wP)


if len(sys.argv) == 9:
    # core.py nlpcc-iccpol-2016.kbqa.testing-data answer kbJson.cleanPre.alias.utf8 outputAP countChar vectorJson.utf8 1 30
    pathInput=sys.argv[1] # nlpcc-iccpol-2016.kbqa.testing-data
    pathOutput=sys.argv[2] # answer
    pathDict=sys.argv[3] # kbJson.cleanPre.alias.utf8
    pathQt=sys.argv[4] # outputAP
    pathCD=sys.argv[5] # countChar
    pathVD=sys.argv[6] # vectorJson.utf8
    qIDstart=int(sys.argv[7]) # 1
    defaultWeightPre=float(sys.argv[8]) # 30
    loadResAndanswerAllQ(pathInput, pathOutput, pathDict, pathQt, pathCD, pathVD, 'utf8', qIDstart, defaultWeightPre)
