from collections import defaultdict


def marginalize(probabilities,index):
    """Given a probability distribution P(X1,...,Xi,...,Xn),
    return the distribution P(X1,...,Xi-1,Xi+1,...,Xn).
    - probabilities: a probability table, given as a map from tuples
      of variable assignments to values
    - index: the value of i.
    """
    res = defaultdict(float)
    for k,v in probabilities.iteritems():
        newk = k[:index]+k[index+1:]
        res[newk] += v
    return res

def marginalize_multiple(probabilities,indices):
    """Safely marginalizes multiple indices"""
    pmarg = probabilities
    for index in reversed(sorted(indices)):
        pmarg = marginalize(pmarg,index)
    return pmarg

def normalize(probabilities):
    """Given an unnormalized distribution, returns a normalized copy that
    sums to 1."""
    vtotal = sum(probabilities.values())
    return dict((k,v/vtotal) for k,v in probabilities.iteritems())

def condition(probabilities,index,value):
    """Given a probability distribution P(X1,...,Xi,...,Xn),
    return the distribution P(X1,...,Xi-1,Xi+1,...,Xn | Xi=v).
    - probabilities: a probability table, given as a map from tuples
      of variable assignments to values
    - index: the value of i.
    - value: the value of v
    """
    res = dict()
    for k,v in probabilities.iteritems():
        if k[index] == value:
            res[k[:index]+k[index+1:]] = v
    return normalize(res)

def naive_bayes(class_probabilities,feature_probabilities,instance):
    """Naive Bayes inference. Given class probabilities P(C) and feature
    conditional probabilities P(Fk|C), compute P(C|F=finstance), where
    - P(C=c) = class_probabilities[c]
    - P(Fk=fk|C=c) = feature_probabilities[feature_name][c][feature_value]
      with feature_name=Fk, feature_value=fk, for all k=1,...,n
    - finstance = instance
    and all features F1,...,Fn are assumed independent, given C.
    """
    dist = {}
    for key, values in class_probabilities.items():
        if key not in dist:
            dist[key] = 1

    normalize = 0

    for key, values in class_probabilities.items():
        for fname,fval in instance.items():
            dist[key] = dist[key] * feature_probabilities[fname][key][fval]
        dist[key] = dist[key]*class_probabilities[key]
        normalize += dist[key]

    for key, values in class_probabilities.items():
        dist[key] = dist[key]/normalize

    return dist

def p1():
    class_probabilities = {'Spam':0.4, 'Not-Spam':0.6}
    feature_probabilities = {
        'f1':{
            'Spam':{0:0.5, 1:0.5},
            'Not-Spam':{0:0.9, 1:0.1}
            },
        'f2':{
            'Spam':{0:0.7, 1:0.3},
            'Not-Spam':{0:0.4, 1:0.6}
            },
        'f3':{
            'Spam':{0:0.4, 1:0.6},
            'Not-Spam':{0:0.99, 1:0.01}
            },
        'f4':{
            'Spam':{0:0.01, 1:0.99},
            'Not-Spam':{0:0.02, 1:0.98}
            }
        }
    instances = {
        "Going to hospital after uncle Larry's twerking incident":{'f1':0,'f2':1,'f3':0,'f4':1},
        "Important message about your password":{'f1':1,'f2':0,'f3':0,'f4':0},
        "**%%V1agr@ and C!al!$ only 25~~! cents49a":{'f1':1,'f2':1,'f3':1,'f4':1},
        "CS270 bug fixes on homework":{'f1':0,'f2':1,'f3':0,'f4':0},
        }
    for subject,features in instances.iteritems():
        pSpam = naive_bayes(class_probabilities,feature_probabilities,features)['Spam']
        print subject,": SPAM probability",pSpam
    return

if __name__=="__main__":
    p1()
