rm -f ict/edu/learning/multiThread/*.class
rm -f ciir/umass/edu/learning/*.class
$JAVA_HOME/bin/javac ict/edu/learning/logisticRankSVM/LogisticRankSVM.java
echo 'compiling is over'
nohup $JAVA_HOME/bin/java ict/edu/learning/logisticRankSVM/LogisticRankSVM -train ../data/\
OHSUMED/OHSUMED/QueryLevelNorm/Fold4/train.txt  -test  ../data/OHSUMED/OHSUMED/QueryLevelNorm/Fold4/test.txt \
-validate  ../data/OHSUMED/OHSUMED/QueryLevelNorm/Fold4/vali.txt -nThread 17 -norm zscore -learningRate 0.00000001 &
echo 'It is done'


