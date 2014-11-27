if [ ! -d "../bin" ]; then
  mkdir ../bin
fi
javac -d ../bin  ict/edu/learning/logisticRankSVM/LogisticRankSVM.java
echo 'compiling is over'
nohup java -cp  ../bin ict/edu/learning/logisticRankSVM/LogisticRankSVM -train ../data/\
OHSUMED/OHSUMED/QueryLevelNorm/Fold1/train.txt  -test  ../data/OHSUMED/OHSUMED/QueryLevelNorm/Fold1/test.txt \
-validate  ../data/OHSUMED/OHSUMED/QueryLevelNorm/Fold1/vali.txt -nThread 17  -maxIterations 100
 -output_interval 3 -learningRate 0.00000001 &
echo 'It is done'



