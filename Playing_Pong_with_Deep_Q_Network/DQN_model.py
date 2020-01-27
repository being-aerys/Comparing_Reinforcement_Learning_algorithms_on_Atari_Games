'''An attempt to solve playing Pong sing DQN following the book by Lapan'''
import torch, time
import torch.nn as nn, numpy as np


#---------------------------ASIDE --> UNPACKING using * ----------------------------------------------------------------------
# def get(shape):
#     print(torch.zeros(*shape)) #(2, 3, 4)
#     print(torch.zeros(1,*shape)) # 2 3 4 --> returns the values after unpacking them
#
# get((1,1,3))
#------------------------------------------------------------------------------------------------------------


class DQN_Model(nn.Module):

    def __init__(self, input_shape, no_of_actions_possbile_in_a_state):
        super(DQN_Model, self).__init__()

        self.conv_part_of_nw = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size= 8, stride= 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64,64, kernel_size=4, stride=1),
            nn.LeakyReLU()

        )

        conv_out_size = self.get_shape_of_an_arbitrary_input_to_the_conv_part(input_shape)

        self.fully_connected_part = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, no_of_actions_possbile_in_a_state)
        )

    '''To make the code generic'''
    def get_shape_of_an_arbitrary_input_to_the_conv_part(self,shape):
        arbitrary_value = self.conv_part_of_nw(torch.zeros(1, *shape))#including 1 because need to pass the batch dimension as well
        #                                                             #when passing an input in pytorch even though the init treats
        #                                                             #as if we passed only three dimensions and treats the batch
        #                                                             #dimension implicitly. However, since we are doing prod later,
        #                                                             #it would not have mattered even if we had not used this 1
        return int(np.prod(arbitrary_value.size())) #Return the product of array elements over a given axis.



    def forward(self, input):
        #Unsqueeze before forwarding
        return (self.fully_connected_part(self.conv_part_of_nw(input).view(input.size()[0],-1))) #size's [0]means batch size









