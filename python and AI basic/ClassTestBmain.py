from ClassTestA import ClassA
def runTester():
    objA=ClassA("inputSomeThing")
    objA.outputTest("output AAA")



if __name__=="__main__":
    print (f"runTester __name__{__name__}")
    runTester()

    #元组
    my_tuple =(1,3,7,"helloWord",[4,5],['a','b'])
    print(my_tuple[0])
    print(my_tuple[2:6])
    print(my_tuple[-1])
    print(my_tuple[0-2][0])