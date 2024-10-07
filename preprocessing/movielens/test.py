from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from functools import reduce


def sum_tags(df):
    l = []
    for i in df:
        a = i.shift(1).apply(lambda x: (x["tags"].split("|"), x["rating"]) if isinstance(x["tags"], str) else [],
                             axis=1).to_list()
        # t1 = reduce(lambda x, y: x + y, a)
        # bb = Counter(t1)
        # bbb = bb.most_common(1)
        # bbb = Counter(reduce(lambda x, y: x[0] + y[0], a))
        ab1=defaultdict(int)
        ab2=defaultdict(float)
        for aa in a:
            if len(aa) ==0:
                continue
            for aaa in aa[0]:
                ab1[aaa] += 1
                ab2[aaa] += aa[1]
        # b = "unknown" if len(bbb) == 0 else bbb[0][0]
        # if not len(a) == 1 and len(a[0]) == 0:
        #     t1 = reduce(lambda x, y: x + y, a)
        #     b = Counter(t1).most_common(1)[0][0]
        # else:
        #     b = "unknown"
        ab3 = {k:ab2[k]/v for k,v in ab1.items()}
        l.append(max(ab3,key=ab3.get) if len(ab3)>0 else "")
    return l
    # return Counter(reduce(lambda x, y: x + y, df.apply(lambda x: x.split("|")))).most_common(1)[0][0]


def main():
    data = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'movieId': [101, 102, 103, 104, 105, 106, 107, 108, 109],
        'timestamp': [1696185600, 1696185700, 1696185800, 1696185900, 1696186000, 1696186100, 1696186200, 1696186300,
                      1696186400],
        'rating': [8, 9, 6, 5, 4, 3, 2, 1, 0],
        'tags': ['action|adventure|comedy', 'comedy', 'drama|thriller|action', 'sci-fi|action|adventure',
                 'action|thriller|horror', 'comedy|drama|romance', 'action|sci-fi|thriller', 'drama|comedy|romance',
                 'sci-fi|drama|action']
    }
    df = pd.DataFrame(data)
    # aa = df.groupby("userId").expanding()["tags"]
    # bb = sum_tags(aa)
    # print(bb)
    b = sum_tags(df.groupby("userId").expanding())
    print(b)


if __name__ == '__main__':
    main()
