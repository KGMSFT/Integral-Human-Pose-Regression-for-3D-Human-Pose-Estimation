from torch import nn
import torch
class Grammer(nn.Module):
    def __init__(self, node=512, joint_num=18):
        super(Grammer,self).__init__()
        self.hidden_size = node 
        self.joint_num = joint_num
        
        #Kinematics Grammar
        self.birnn_kin_1 = nn.RNN(input_size=joint_num*3, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc_kin_1 = nn.Linear(self.hidden_size * 2, 3)

        self.birnn_kin_2 = nn.RNN(input_size=joint_num*3, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc_kin_2 = nn.Linear(self.hidden_size * 2, 3)
        
        self.birnn_kin_3 = nn.RNN(input_size=joint_num*3, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc_kin_3 = nn.Linear(self.hidden_size * 2, 3)

        self.birnn_kin_4 = nn.RNN(input_size=joint_num*3, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc_kin_4 = nn.Linear(self.hidden_size * 2, 3)

        self.birnn_kin_5 = nn.RNN(input_size=joint_num*3, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc_kin_5 = nn.Linear(self.hidden_size * 2, 3)

        #Symmetry Grammer
        self.birnn_sym_1 = nn.RNN(input_size=joint_num*3, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc_sym_1 = nn.Linear(self.hidden_size * 2, 3)
        
        self.birnn_sym_2 = nn.RNN(input_size=joint_num*3, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc_sym_2 = nn.Linear(self.hidden_size * 2, 3)
        
        #Coordination Grammer
        self.birnn_crd_1 = nn.RNN(input_size=joint_num*3, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc_crd_1 = nn.Linear(self.hidden_size * 2, 3)
        
        self.birnn_crd_2 = nn.RNN(input_size=joint_num*3, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc_crd_2 = nn.Linear(self.hidden_size * 2, 3)


    def forward(self, x):
        x = x.repeat((1, self.joint_num))
        x = x.view(-1, self.joint_num, self.joint_num * 3)
        chain1 = self.getRightLeg(x)
        chain2 = self.getLeftLeg(x)
        chain3 = self.getRightArm(x)
        chain4 = self.getLeftArm(x)
        chain5 = self.getHeadSpine(x)
        chain6 = self.getUpperLimb(x)
        chain7 = self.getLowerLimb(x)
        chain8 = self.getLeftArmRightLeg(x)
        chain9 = self.getRightArmLeftLeg(x)

        # Kinematics Grammer
        output1, _ = self.birnn_kin_1(chain1)

        output1 = self.fc_kin_1(output1)
        output1 = output1.view((-1, 4 * 3))

        output2, _ = self.birnn_kin_2(chain2)
        output2 = self.fc_kin_2(output2)
        output2 = output2.view((-1, 4 * 3))

        output3, _ = self.birnn_kin_3(chain3)
        output3 = self.fc_kin_3(output3)
        output3 = output3.view((-1, 4 * 3))

        output4, _ = self.birnn_kin_4(chain4)
        output4 = self.fc_kin_4(output4)
        output4 = output4.view((-1, 4 * 3))

        output5, _ = self.birnn_kin_5(chain5)
        output5 = self.fc_kin_5(output5)
        output5 = output5.view((-1, 4*3))

        # Symmetry Grammer
        output6, _ = self.birnn_sym_1(chain6)
        output6 = self.fc_sym_1(output6)
        output6 = output6.view((-1, 7 * 3))

        output7, _ = self.birnn_sym_2(chain7)
        output7 = self.fc_sym_2(output7)
        output7 = output6.view((-1, 7 * 3))

        # Coordination Grammer
        output8, _ = self.birnn_crd_1(chain8)
        output8 = self.fc_crd_1(output8)
        output8 = output8.view((-1, 8 * 3))

        output9, _ = self.birnn_crd_2(chain9)
        output9 = self.fc_crd_2(output9)
        output9 = output9.view((-1, 8 * 3))
        
        joint_16 = self.merge_func([output1, output2, output3, output4, output5, output6, output7, output8, output9], x)
        return joint_16


    def getRightLeg(self, x):
        x1 = x.permute(1, 2, 0)
        x1 = x1[[0,1,2,3],:,:]
        x1 = x1.permute(2, 0, 1)
        return x1

    def getLeftLeg(self, x):
        x1 = x.permute(1, 2, 0)
        x1 = x1[[0,4,5,6],:,:]
        x1 = x1.permute(2, 0, 1)
        return x1



    def getRightArm(self, x):
        x1 = x.permute(1, 2, 0)
        x1 = x1[[8,14,15,16],:,:]
        x1 = x1.permute(2, 0, 1)
        return x1

    def getLeftArm(self, x):
        x1 = x.permute(1, 2, 0)
        x1 = x1[[8,11,12,13],:,:]
        x1 = x1.permute(2, 0, 1)
        return x1     


    #chain5
    def getHeadSpine(self, x):
        x1 = x.permute(1, 2, 0)
        x1 = x1[[0,7,8,10],:,:]
        x1 = x1.permute(2, 0, 1)
        return x1

    # chain6
    def getUpperLimb(self, x):
        x1 = x.permute(1, 2, 0)
        x1 = x1[[8,11,12,13,14,15,16],:,:]
        x1 = x1.permute(2, 0, 1)
        return x1   
    # chain7
    def getLowerLimb(self, x):
        x1 = x.permute(1, 2, 0)
        x1 = x1[[0,1,2,3,4,5,6],:,:]
        x1 = x1.permute(2, 0, 1)
        return x1   
    # chain8
    def getLeftArmRightLeg(self, x):
        x1 = x.permute(1,2,0)
        x1 = x1[[8,11,12,13,0,1,2,3]]
        x1 = x1.permute(2, 0, 1)
        return x1
    # chain9
    def getRightArmLeftLeg(self, x):
        x1 = x.permute(1,2,0)
        x1 = x1[[8,14,15,16,0,4,5,6]]
        x1 = x1.permute(2, 0, 1)
        return x1
    def merge_func(self, inputs, x):
        rhip = (inputs[0][:,3:6] + inputs[6][:,3:6] + inputs[7][:,15:18]) / 3
        rhip = rhip.view(-1, 3)
        rknee = (inputs[0][:,6:9] + inputs[6][:,6:9] + inputs[7][:,18:21]) / 3
        rknee = rknee.view(-1, 3)
        rank = (inputs[0][:,9:] + inputs[6][:,9:12] + inputs[7][:,21:]) / 3

        lhip = (inputs[1][:,3:6] + inputs[6][:,12:15] + inputs[8][:,15:18]) / 3
        lhip = lhip.view(-1, 3)
        lknee = (inputs[1][:,6:9] + inputs[6][:,15:18] + inputs[8][:,18:21]) / 3
        lknee = lknee.view(-1, 3)
        lank = (inputs[1][:,9:] + inputs[6][:,18:] + inputs[8][:,21:]) / 3
        lank = lank.view(-1, 3)

        rshoudler = (inputs[2][:,3:6] + inputs[5][:,12:15] + inputs[8][:,3:6]) / 3
        rshoudler = rshoudler.view(-1, 3)
        relbow = (inputs[2][:,6:9] + inputs[5][:,15:18] + inputs[8][:,6:9]) / 3
        relbow = relbow.view(-1, 3)
        rwrist = (inputs[2][:,9:] + inputs[5][:,18:] + inputs[8][:,9:12]) / 3
        rwrist = rwrist.view(-1, 3)

        lshoudler = (inputs[3][:,3:6] + inputs[5][:,3:6] + inputs[7][:,3:6]) / 3
        lshoudler = lshoudler.view(-1, 3)
        lelbow = (inputs[3][:,6:9] + inputs[5][:,6:9] + inputs[7][:,6:9]) / 3
        lelbow = lelbow.view(-1, 3)
        lwrist = (inputs[3][:,9:] + inputs[5][:,9:12] + inputs[7][:,9:12]) / 3
        lwrist = lwrist.view(-1, 3)
        
        pelvis = (inputs[0][:,0:3] + inputs[1][:,0:3] + inputs[4][:,0:3] + inputs[6][:,0:3] +  inputs[7][:,12:15] + inputs[8][:,12:15]) / 6
        pelvis = pelvis.view(-1, 3)
        spine = (inputs[4][:,3:6])
        spine = spine.view(-1, 3)
        neck   = (inputs[2][:,0:3] + inputs[3][:,0:3] + inputs[4][:,6:9] + inputs[5][:,0:3] + inputs[7][:,0:3] + inputs[8][:,0:3]) / 6
        neck = neck.view(-1, 3)
        head = inputs[4][:,9:]
        head = head.view(-1, 3)
        joint_18 = torch.cat((pelvis, rhip, rknee, rank, lhip, lknee, lank, spine, neck, x[:,0,27:30], head, lshoudler, lelbow, lwrist, rshoudler, relbow, rwrist, x[:,0,-3:]), 1)
        joint_18 = joint_18.view(-1, self.joint_num, 3)
        return joint_18