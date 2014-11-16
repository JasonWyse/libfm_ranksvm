$JAVA_HOME/bin/javac -d bin/  ict/edu/learning/test/Test.java
echo 'compiling is over'
nohup $JAVA_HOME/bin/java ../bin/ict/edu/learning/test/Test -filename matrixV/matrixV.txt  -train ../data/\
OHSUMED/OHSUMED/QueryLevelNorm/Fold1/train.txt  -test  ../data/OHSUMED/OHSUMED/QueryLevelNorm/Fold1/test.txt \
-validate   ../data/OHSUMED/OHSUMED/QueryLevelNorm/Fold1/vali.txt -nThread 17 -norm zscore -learningRate 0.00000001 &
echo 'It is done'


