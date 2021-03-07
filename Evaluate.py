import pickle
import os
import sys


def evaluavtion_triple(testresult):
    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.

    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        predictrightnum, predictnum, rightnum = count_sentence_triple_num(ptag, ttag)
        total_predict_right += predictrightnum
        total_predict += predictnum
        total_right += rightnum
    # print(total_predict_right,total_predict,total_right)
    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2*P*R)/float(P+R) if P != 0 else 0

    return P, R, F

    # total_predict_right1 = 0.
    # total_predict1 = 0.
    # total_predict_right2 = 0.
    # total_predict2 = 0.
    # total_predict_right = 0.
    # total_predict = 0.
    # total_right = 395.
    # for sent in testresult:
    #     ptag = sent[0]
    #     ttag = sent[result]
    #     rightnum, predictnum, rightnume1, rightnume2, predictnume1, predictnume2 \
    #         = count_predict_right_num(ptag, ttag)
    #     total_predict_right1 += rightnume1
    #     total_predict1 += predictnume1
    #     total_predict_right2 += rightnume2
    #     total_predict2 += predictnume2
    #     total_predict_right += rightnum
    #     total_predict += predictnum
    # print(total_predict_right1,total_predict1,total_predict_right2,total_predict2,total_predict_right,total_predict)
    #
    # P1 = total_predict_right1 / float(total_predict1) if total_predict1 != 0 else 0
    # R1 = total_predict_right1 / float(total_right)
    # F1 = (2*P1*R1)/float(P1+R1) if P1 != 0 else 0
    # P2 = total_predict_right2 / float(total_predict2) if total_predict2 != 0 else 0
    # R2 = total_predict_right2 / float(total_right)
    # F2 = (2*P2*R2)/float(P2+R2) if P2 != 0 else 0
    # P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    # R = total_predict_right / float(total_right)
    # F = (2*P*R)/float(P+R) if P != 0 else 0
    # return P1, R1, F1, P2, R2, F2, P, R, F


def count_sentence_triple_num(ptag, ttag):
    predict_rmpair = tag_to_triple_index(ptag)  # transfer the predicted tag sequence to triple index
    right_rmpair = tag_to_triple_index(ttag)
    predict_right_num = 0       # the right number of predicted triple
    predict_num = 0     # the number of predicted triples
    right_num = 0

    for type in predict_rmpair:
        eelist = predict_rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        # predict_num += min(len(e1), len(e2))
        #
        # if predict_num > result:
        #     predict_num += result
        if min(len(e1), len(e2)) >= 1:
            predict_num += 1
        if predict_num > 1:
            predict_num = 1

        if right_rmpair.__contains__(type):
            reelist = right_rmpair[type]
            re1 = reelist[0]
            re2 = reelist[1]
            for i in range(0, min(min(len(e1), len(e2)), min(len(re1), len(re2)))):
                if e1[i][0] == re1[i][0] and e1[i][1] == re1[i][1] and e2[i][0] == re2[i][0] \
                        and e2[i][1] == re2[i][1]:
                    predict_right_num += 1
        else:
            a = predict_rmpair.__contains__('/location/location/contains') or\
                predict_rmpair.__contains__('/location/country/administrative_divisions') or\
                predict_rmpair.__contains__('/location/administrative_division/country') or\
                predict_rmpair.__contains__('/location/country/capital')
            b = right_rmpair.__contains__('/location/location/contains') or\
                right_rmpair.__contains__('/location/country/administrative_divisions') or\
                right_rmpair.__contains__('/location/administrative_division/country')
            c = predict_rmpair.__contains__('/business/company_shareholder/major_shareholder_of') or\
                predict_rmpair.__contains__('/business/person/company')
            d = right_rmpair.__contains__('/business/person/company')
            if (a and b) or (c and d):
                if right_rmpair.__contains__('/location/location/contains'):
                    reelist = right_rmpair['/location/location/contains']
                    re1 = reelist[0]
                    re2 = reelist[1]
                    for i in range(0, min(min(len(e1), len(e2)), min(len(re1), len(re2)))):
                        if e1[i][0] == re1[i][0] and e1[i][1] == re1[i][1] and e2[i][0] == re2[i][0] \
                                and e2[i][1] == re2[i][1]:
                            predict_right_num += 1
                if right_rmpair.__contains__('/location/country/administrative_divisions'):
                    reelist = right_rmpair['/location/country/administrative_divisions']
                    re1 = reelist[0]
                    re2 = reelist[1]
                    for i in range(0, min(min(len(e1), len(e2)), min(len(re1), len(re2)))):
                        if e1[i][0] == re1[i][0] and e1[i][1] == re1[i][1] and e2[i][0] == re2[i][0] \
                                and e2[i][1] == re2[i][1]:
                            predict_right_num += 1
                if right_rmpair.__contains__('/location/administrative_division/country'):
                    reelist = right_rmpair['/location/administrative_division/country']
                    re1 = reelist[0]
                    re2 = reelist[1]
                    for i in range(0, min(min(len(e1), len(e2)), min(len(re1), len(re2)))):
                        if e1[i][0] == re1[i][0] and e1[i][1] == re1[i][1] and e2[i][0] == re2[i][0] \
                                and e2[i][1] == re2[i][1]:
                            predict_right_num += 1
                if right_rmpair.__contains__('/business/person/company'):
                    reelist = right_rmpair['/business/person/company']
                    re1 = reelist[0]
                    re2 = reelist[1]
                    for i in range(0, min(min(len(e1), len(e2)), min(len(re1), len(re2)))):
                        if e1[i][0] == re1[i][0] and e1[i][1] == re1[i][1] and e2[i][0] == re2[i][0] \
                                and e2[i][1] == re2[i][1]:
                            predict_right_num += 1

    for type in right_rmpair:
        eelist = right_rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        right_num += min(len(e1), len(e2))
    return predict_right_num, predict_num, right_num


def tag_to_triple_index(ptag):
    rmpair = {}
    for i in range(0, len(ptag)):
        tag = ptag[i]
        if not tag.__eq__("O") and not tag.__eq__(""):
            type_e = tag.split("__")
            if not rmpair.__contains__(type_e[0]):
                eelist = []
                e1 = []
                e2 = []
                if type_e[1].__contains__("result"):
                    if type_e[1].__contains__("S"):
                        e1.append((i, i+1))
                    elif type_e[1].__contains__("B"):
                        j = i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("result") and (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i, i+1))
                    elif type_e[1].__contains__("B"):
                        j = i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e2.append((i, j))
                eelist.append(e1)
                eelist.append(e2)
                rmpair[type_e[0]] = eelist
            else:
                eelist = rmpair[type_e[0]]
                e1 = eelist[0]
                e2 = eelist[1]
                if type_e[1].__contains__("result"):
                    if type_e[1].__contains__("S"):
                        e1.append((i, i+1))
                    elif type_e[1].__contains__("B"):
                        j = i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("result") and (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i, i+1))
                    elif type_e[1].__contains__("B"):
                        j = i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e2.append((i, j))
                eelist[0] = e1
                eelist[1] = e2
                rmpair[type_e[0]] = eelist
    return rmpair


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def count_predict_right_num(ptag, ttag):
    rmpair = {}
    # masktag = np.zeros(len(ptag))
    for i in range(0, len(ptag)):
        tag = ptag[i]
        if not tag.__eq__("O") and not tag.__eq__(""):
            type_e = tag.split("__")
            if not rmpair.__contains__(type_e[0]):
                eelist = []
                e1 = []
                e2 = []
                if type_e[1].__contains__("result"):
                    if type_e[1].__contains__("S"):
                        e1.append((i, i+1))
                    elif type_e[1].__contains__("B"):
                        j = i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("result") and (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i, i+1))
                    elif type_e[1].__contains__("B"):
                        j = i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e2.append((i, j))
                eelist.append(e1)
                eelist.append(e2)
                rmpair[type_e[0]] = eelist
            else:
                eelist = rmpair[type_e[0]]
                e1 = eelist[0]
                e2 = eelist[1]
                if type_e[1].__contains__("result"):
                    if type_e[1].__contains__("S"):
                        e1.append((i, i+1))
                    elif type_e[1].__contains__("B"):
                        j = i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("result") and (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i, i+1))
                    elif type_e[1].__contains__("B"):
                        j = i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e2.append((i, j))
                eelist[0] = e1
                eelist[1] = e2
                rmpair[type_e[0]] = eelist

    right_rmpair = tag_to_triple_index(ttag)

    rightnum = 0
    rightnume1 = 0
    rightnume2 = 0
    predictnum = 0
    predictnume1 = 0
    predictnume2 = 0
    global re1
    global re2
    for type in rmpair:
        eelist = rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        if min(len(e1), len(e2)) >= 1:
            predictnum += 1
        if predictnum > 1:
            predictnum = 1

        if len(e1) > 0:
            predictnume1 += 1
        if predictnume1 > 1:
            predictnume1 = 1
        if len(e2) > 0:
            predictnume2 += 1
        if predictnume2 > 1:
            predictnume2 = 1

        for name in right_rmpair.keys():
            reelist = right_rmpair[name]
            re1 = reelist[0]
            re2 = reelist[1]

        for i in range(0, min(min(len(e1), len(e2)), min(len(re1), len(re2)))):
            truemark = 0
            truemarke1 = 0
            truemarke2 = 0

            if e1[i][0] == re1[i][0] and e1[i][1] == re1[i][1] and e2[i][0] == re2[i][0] \
                    and e2[i][1] == re2[i][1]:
                truemark = 1

            for j in range(e1[i][0], e1[i][1]):
                if e1[i][0] > 0 and ttag[e1[i][0]].__contains__(type) and\
                        e1[i][1] < len(ttag) and ttag[e1[i][1]-1].__contains__(type):
                    truemarke1 = 1
                else:
                    a = ptag[j].__contains__('/location/location/contains') or \
                        ptag[j].__contains__('/location/country/administrative_divisions') or \
                        ptag[j].__contains__('/location/administrative_division/country') or \
                        ptag[j].__contains__('/location/country/capital')
                    b = ttag[j].__contains__('/location/location/contains') or \
                        ttag[j].__contains__('/location/country/administrative_divisions') or \
                        ttag[j].__contains__('/location/administrative_division/country')
                    c = ptag[j].__contains__('/business/company_shareholder/major_shareholder_of') or \
                        ptag[j].__contains__('/business/person/company')
                    d = ttag[j].__contains__('/business/person/company')
                    if (a and b) or (c and d):
                        truemarke1 = 1
                if not ttag[j].__contains__("result"):
                    truemarke1 = 0
                    break
            for j in range(e2[i][0], e2[i][1]):
                if e2[i][0] > 0 and ttag[e2[i][0]].__contains__(type)and\
                        e2[i][1] < len(ttag) and ttag[e2[i][1]-1].__contains__(type):
                    truemarke2 = 1
                else:
                    a = ptag[j].__contains__('/location/location/contains') or \
                        ptag[j].__contains__('/location/country/administrative_divisions') or \
                        ptag[j].__contains__('/location/administrative_division/country') or \
                        ptag[j].__contains__('/location/country/capital')
                    b = ttag[j].__contains__('/location/location/contains') or \
                        ttag[j].__contains__('/location/country/administrative_divisions') or \
                        ttag[j].__contains__('/location/administrative_division/country')
                    c = ptag[j].__contains__('/business/company_shareholder/major_shareholder_of') or \
                        ptag[j].__contains__('/business/person/company')
                    d = ttag[j].__contains__('/business/person/company')
                    if (a and b) or (c and d):
                        truemarke2 = 1
                if not ttag[j].__contains__("2"):
                    truemarke2 = 0
                    break
            if truemark == 1:
                rightnum += 1
            if truemarke1 == 1:
                rightnume1 += 1
            if truemarke2 == 1:
                rightnume2 += 1
    return rightnum, predictnum, rightnume1, rightnume2, predictnume1, predictnume2


if __name__ == "__main__":

    Pre = 0
    Rec =0
    F1 = 0
    for i in range(1, 28):
            resultname = "./data/demo/result1/result-{}".format(i * 2)
            testresult = pickle.load(open(resultname, 'rb'))
            P, R, F = evaluavtion_triple(testresult)
            if F > F1:
                Pre = P
                Rec = R
                F1 = F
            print('Epoch: [{}/{}], P R F {:.4f} {:.4f} {:.4f}'.format(i * 2, 100, P, R, F))
    print(Pre, Rec, F1)


            # P1, R1, F1, P2, R2, F2, P, R, F = evaluavtion_triple(testresult)
            # print('Epoch: [{}/{}], P1 R1 F1 P2 R2 F2 P R F {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(i * 2, 100, P1, R1, F1, P2, R2, F2, P, R, F))
