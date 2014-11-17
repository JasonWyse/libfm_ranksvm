if [ ! -d "../bin" ]; then
  mkdir ../bin
fi
$JAVA_HOME/bin/javac -d ../bin  ict/edu/learning/logisticRankSVM/LogisticRankSVM.java
echo 'compiling is over'
nohup $JAVA_HOME/bin/java -cp  ../bin ict/edu/learning/logisticRankSVM/LogisticRankSVM -train ../data/\
OHSUMED/OHSUMED/QueryLevelNorm/Fold3/train.txt  -test  ../data/OHSUMED/OHSUMED/QueryLevelNorm/Fold3/test.txt \
-validate  ../data/OHSUMED/OHSUMED/QueryLevelNorm/Fold3/vali.txt -nThread 17 -norm zscore -learningRate 0.00000001 &
echo 'It is done'



