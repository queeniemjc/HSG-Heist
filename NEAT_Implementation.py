from __future__ import print_function
import os
import neat
import random
import numpy as np
import pickle
import math
import multiprocessing

runs_per_net = 20
num_of_generations = 700
AddWalls = False

#Game related variables
mapwidth = 30
mapheight = 20
run = True
score = 0
GameOver = False
IsMenuScreen = True

# Student stuff
StudentVec = np.array([0,5])
StudentChange = np.array([1,0])
BestStudent = None

# Sturm stuff
SturmVec = np.array([26,19])
SturmChange = np.array([-1,0])
BestSturm = None

# Token stuff
TokenVec = np.array([3,0])
ListOfTokenSpawnPoints = [np.array([10,2]),np.array([12,4]),np.array([21,6]),np.array([5,6]),np.array([5,11]), np.array([11,8]), np.array([16,8]), np.array([25,8]), np.array([10,9]), np.array([16,13]), np.array([19,11]), np.array([24,19])]

generationcount = 0

map2 = [
    "000000000000000000000000000000",    
    "000000000010000000000000000000",
    "000000000010000000000000000000",
    "000000000010000000000000000000",
    "000000001111111111111111111110",
    "000000001111111000010000000000",
    "000001111110001000011100000000",        
    "000001111110001000000100000000",   
    "000001110011111111000111111110",
    "000001110011111111000100000000",                         
    "000001110011111111111100000000",                    
    "000001111111111111111100000000",
    "000000001110000011001100000000",
    "000000001110000011001100000000",
    "000000001110000001001100000000",
    "000000001110000001001100000000", 
    "000000000100000001111100000000",              
    "000000000100000001000100000000",   
    "000000000100000001000100000000",
    "000000000111111111111111111110"                  
]
GameMap = [
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111"
]



def IsValid(Vec):
    global GameMap
    global mapheight
    global mapwidth
    if Vec[0] < 0:
        return False
    if Vec[0] > mapwidth - 1:
        return False
    if Vec[1] < 0:
        return False
    if Vec[1] > mapheight - 1:
        return False
    
    #Is the movement to a valid space? (if statement needed to make sure indexing is done within the range of the map)
    if GameMap[Vec[1]][Vec[0]] == "0":
            return False
    return True
    
def Move(Vec, Change, OpponentVec, genome):
    ## First check if the move is legal -> WITHIN BOUNDS, NOT ON INVALID SPACE, NOT ON STURM
    global mapwidth
    global mapheight
    global GameMap
    CanMove = True
    run = True #In case we want to abort the run after some movement failure
    #Is the movement within the bounds of the map? (a 30x20 grid)
    NextVec = Vec + Change
    CanMove = IsValid(NextVec)
    """
    while not(CanMove):
        #Is this movement legal?
        NextVec = Vec + Change
        CanMove = IsValid(NextVec)

        #If you hit a wall, move randomly to a valid space
        PossibleMovementList = []
        BasisVecList = [(1,0), (0,1), (-1,0), (0,-1)]
        for i in range(4):
            if IsValid(BasisVecList[i] + Vec):
                PossibleMovementList.append(BasisVecList[i])
        
        Rand = random.randint(0, len(PossibleMovementList) - 1)
        Change = PossibleMovementList[Rand]
        if IsValid(Vec + Change):
            CanMove = True"""
            
            
    #Is the movement to Sturm? - Useful failsafe
    if NextVec[0] == OpponentVec[0] and NextVec[1] == OpponentVec[1]:
        CanMove = False
    if CanMove: 
        Vec[0] = Vec[0] + Change[0]
        Vec[1] = Vec[1] + Change[1]
    return (Vec, genome.fitness, run)

def TokenHandler(LocalStudentVec, LocalTokenVec, LocalScore):
    # If you catch the toxen, then a new token needs to be initialized - as well as changing the picture Token is using - and changing score
    HasWonToken = False
    if LocalStudentVec[0] == LocalTokenVec[0] and LocalStudentVec[1] == LocalTokenVec[1]:
        LocalScore = LocalScore + 1
        LocalTokenVec = ShuffleToken(LocalTokenVec)
        HasWonToken = True
    

    #In case the token spawns at the same place as before, we want it to go somewhere else instead.
    while LocalStudentVec[0] == LocalTokenVec[0] and LocalStudentVec[1] == LocalTokenVec[1]:
        LocalTokenVec = ShuffleToken(LocalTokenVec)
    return (LocalTokenVec, LocalScore, HasWonToken)

def ShuffleToken(LocalTokenVec):
    #Choose one of the vectors for the new Token at random
    global ListOfTokenSpawnPoints
    randn = random.randint(0, len(ListOfTokenSpawnPoints) - 1)
    LocalTokenVec = ListOfTokenSpawnPoints[randn]
    return LocalTokenVec

def IsCaught(LocalStudentVec, LocalSturmVec):
    if np.linalg.norm(LocalStudentVec - LocalSturmVec) <= 1.3:
        return True
    else:
        return False

def eval_genomes_of_student(genomes, config):
    global StudentVec
    global StudentChange
    global SturmVec
    global SturmChange
    global BestSturm
    global generationcount
    global GameMap
    global map2
    if BestSturm != None:
        BestSturmNet = neat.nn.FeedForwardNetwork.create(BestSturm, config)
    GameMap = ReintroduceMap(generationcount, map2, GameMap)
    for genome_id, genome in genomes:
        #Create local copies so that the parallel simulations do not accidentally change each others' global variable.
        LocalScore = 0
        

        randn = random.randint(0, 3)
        SpawnLocations = [np.array([13,9]), np.array([14,9]), np.array([15,9]), np.array([9,13])]
        LocalStudentVec = SpawnLocations[randn]
        LocalStudentChange = StudentChange
        LocalSturmVec = SturmVec 
        LocalSturmChange = SturmChange

        LocalTokenVec = TokenVec
        
        genome.fitness = 0
        LocalRoundCount = 0

        StudentNet = neat.nn.FeedForwardNetwork.create(genome, config)

        ### Play the whole game here with the better Sturm

        ### MOVEMENT
        run = True
        LocalTokenVec = ShuffleToken(LocalTokenVec)

        while run:
            #print(LocalStudentVec, LocalSturmVec, LocalTokenVec)
            LocalStudentChange = StudentNet.activate(GiveInput(LocalStudentVec, LocalSturmVec, LocalTokenVec))
            #print(LocalStudentChange)
            #Normalize NN Output - to which basis vector is (x,y) leaning the most?
            if(abs(LocalStudentChange[0]) >= abs(LocalStudentChange[1])):
                LocalStudentChange[1] = 0
                if(LocalStudentChange[0] > 0):
                    LocalStudentChange[0] = 1
                else:
                    LocalStudentChange[0] = -1
            else:
                LocalStudentChange[0] = 0
                if(LocalStudentChange[1] > 0):
                    LocalStudentChange[1] = 1
                else:
                    LocalStudentChange[1] = -1
            
         
          
            (LocalStudentVec, genome.fitness, run) = Move(LocalStudentVec, LocalStudentChange, LocalSturmVec, genome)
            if BestSturm != None:
                LocalSturmChange = BestSturmNet.activate(GiveInputSturm(LocalSturmVec, LocalStudentVec))
                ## Turn BestSturmNet Output discrete
                if(abs(LocalSturmChange[0]) > abs(LocalSturmChange[1])):
                    LocalSturmChange[1] = 0
                    if(LocalSturmChange[0] > 0):
                        LocalSturmChange[0] = 1
                    else:
                        LocalSturmChange[0] = -1
                else:
                    LocalSturmChange[0] = 1
                    if(LocalSturmChange[1] > 0):
                        LocalSturmChange[1] = 1
                    else:
                        LocalSturmChange[1] = -1
                #(LocalSturmVec, [], run) = Move(LocalSturmVec, LocalSturmChange, LocalStudentVec, genome)
            (LocalTokenVec, LocalScore, HasWonToken) = TokenHandler(LocalStudentVec, LocalTokenVec, LocalScore)

            #Update genome fitness, punish for being slow
            if HasWonToken:
                genome.fitness += 1 + math.exp(-LocalRoundCount/40)
            if LocalScore > 10: #Local Student now good enough, could continue indefinitely
                run = False
            if LocalRoundCount >= 500 + 50*LocalScore: #Terminate if it really doesn't figure it out
                run = False
            #run = IsCaught(LocalStudentVec, LocalSturmVec)

            """print("start")
            print(LocalTokenVec)
            print(type(LocalTokenVec))
            print(LocalStudentVec)
            print(type(LocalStudentVec))"""
   
                
            LocalRoundCount += 1
    generationcount += 1

def ReintroduceMap(generationcount, newmap, emptymap):
    #Used to slowly ease in the NN to the addition of invalid spaces
    # (e.g. walls) so that it can adapt its pathways
    #_print_map(newmap)
    #_print_map(emptymap)
    global AddWalls
    if(generationcount >= 100 and AddWalls):
        DifferenceList = []
        RandN = random.randint(0,600) 
        for x in range(30): 
            for y in range(20): 
                if(newmap[y][x] != emptymap[y][x]):
                    DifferenceList.append([x,y])
        if len(DifferenceList) != 0:
            index = RandN % (len(DifferenceList))
            XRep, YRep = DifferenceList[index] 
            leftside = emptymap[YRep][0:XRep]
            rightside = emptymap[YRep][XRep+1:]
            emptymap[YRep] = leftside + newmap[YRep][XRep] + rightside 
        return emptymap
    return emptymap

def GiveInput(LocalStudentVec, LocalSturmVec, LocalTokenVec):
    # This function creates the input vector given to the Sturm/Student NN: A 5x5 grid around Sturm/Student (+ Distance to opponent) + Distance to token
    # A valid space is encoded as a 1, an invalid space as a 0
    InputVec = []
    HostVec = LocalStudentVec
    global GameMap
    global mapwidth
    global mapheight
    list = [-2,-1, 0, 1, 2]
    
    # First insert the 3x3 grid:
    
    for x in list:
        for y in list:
            absx = x + HostVec[0]
            absy = y + HostVec[1]
          
            ### if x == y == 0, then this is where the AI is standing - and Sturm AI/Student AI doesn't need to know it is on a valid space.
            if not(x == 0 and y == 0):
                #if absx and absy are outside the grid accessing map will give an error - 
                #so if they are outside we can handle them as invalid spaces
                if (absx >= 0 and absy >= 0 and absx <= mapwidth - 1 and absy <= mapheight - 1):
                    if GameMap[absy][absx] == '0':
                        InputVec.append(0)
                    else:
                        InputVec.append(1)
                else: 
                    InputVec.append(0)
    #The following block inputs some extra stuff to the 3x3 grid to extend SturmAI/StudentAI's range - tiles outside the grid are once again invalid
    """
    shift = [-2,2]
    for x in shift:
        absx = x + HostVec[0]
        absy = y + HostVec[1]
        if(absx >= 0 and absy >= 0 and absx <= mapwidth - 1 and absy <= mapheight -1):
            if map[absy][absx] == '0':
                InputVec.append(0)
            else:
                InputVec.append(1)
        else:
            InputVec.append(0)
    for y in shift:
        absx = x + HostVec[0]
        absy = y + HostVec[1]
        if(absx >= 0 and absy >= 0 and absx <= mapwidth - 1 and absy <= mapheight -1):
            if map[absy][absx] == '0':
                InputVec.append(0)
            else:
                InputVec.append(1)
        else:
            InputVec.append(0)
    """
    # Distance of Sturm to Student
    """
    InputVec.append(OpponentVec[0] - HostVec[0])
    InputVec.append(OpponentVec[1] - HostVec[1])
    """
   

    # Distance of Student to Token
    InputVec.append(LocalTokenVec[0] - HostVec[0]) 
    InputVec.append(LocalTokenVec[1] - HostVec[1])
    if(len(InputVec) != 26):
        print("something's wrong")
    return InputVec

def GiveInputSturm(SturmVec, StudentVec):
    InputVec = []
    # This function creates the input vector given to the Sturm/Student NN: A 3x3 grid around Sturm/Student + 4 further tiles + Distance to opponent + Distance to token
    # A valid space is encoded as a 1, an invalid space as a 0
    HostVec = SturmVec
    OpponentVec = StudentVec
    global GameMap
    global mapwidth
    global mapheight
    list = [-1, 0, 1]
    
    # First insert the 3x3 grid:
    for x in list:
        for y in list:
            
            absx = x + HostVec[0]
            absy = y + HostVec[1]
          
            ### if x == y == 0, then this is where Sturm is standing - and Sturm AI/Student AI doesn't need to know it is on a valid space.
            if not(x == 0 and y == 0):
                #if absx and absy are outside the grid accessing GameMap will give an error - 
                #so if they are outside we can handle them as invalid spaces
                if (absx >= 0 and absy >= 0 and absx <= mapwidth - 1 and absy <= mapheight - 1):
                    if GameMap[absy][absx] == '0':
                        InputVec.append(0)
                    else:
                        InputVec.append(1)
                else: 
                    InputVec.append(0)
           

    #The following block inputs some extra stuff to the 3x3 grid to extend SturmAI/StudentAI's range - tiles outside the grid are once again invalid
    shift = [-3,3]
    for x in shift:
        for y in shift:
            absx = x + HostVec[0]
            absy = y + HostVec[1]
            if(absx >= 0 and absy >= 0 and absx <= mapwidth - 1 and absy <= mapheight -1):
                if GameMap[absy][absx] == '0':
                    InputVec.append(0)
                else:
                    InputVec.append(1)
            else:
                InputVec.append(0)

    # Distance of Sturm to Student
    InputVec.append(OpponentVec[0] - HostVec[0])
    InputVec.append(OpponentVec[1] - HostVec[1])

    """# Distance of Sturm to Token (to intercept the Student)
    InputVec.append(TokenVec[0] - HostVec[0])
    InputVec.append(TokenVec[1] - HostVec[1])""" 

    if(len(InputVec) != 16):
        print("something's wrong")
    return InputVec

def run_student(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    #pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes_of_student)
    # Run for up to 300 generations.
    winner = p.run(eval_genomes_of_student, num_of_generations)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()
    
    global BestStudent
    BestStudent = winner

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    #visualize.draw_net(config, winner, True)
    #node_names = {-1:'A', -2: 'B', 0:'A XOR B'}

    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes_of_student, 10)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run_student(config_path)



