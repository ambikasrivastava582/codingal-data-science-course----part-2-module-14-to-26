score = 0
def q1():
    q1 = "What do we call the first layer of this neural network?"
    a1 = input("Enter your answer : ")
    a1 = a1.lower()
    if a1=='input layer':
     s = 1
    else:
      s=0
    return s
score += q1()
def q2():
      q2 = "How many number of neurons can be there in the first layer of this neural network?"
      a2 = int(input("Enter your answer : "))
      if a2==8:
        s = 1
      else:
         s=0
      return s