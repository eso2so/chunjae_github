from pipeline.extract import df
from pipeline.controller import etl
import sys

try:
    etl
    print('실행되었습니다')
except Exception as e:
    print(e)
    sys.exit
sys.exit(0)












# df = extract()
# data = preprocess(df)
# model = MyModel()
# model.fit(data)




