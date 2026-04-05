import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.functional as functional

class Net(nn.Module):
    def __init__(self, angRes_in, angRes_out):
        super(Net, self).__init__()
        channels = 64
        n_group=2
        n_block = 4
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.init_fe=init_FE_Block(angRes_in,channels)
        self.DeepConvNet = CascadedDeepConvGroup(n_group,n_block, angRes_in, channels)       
        self.cubic=Interpolation(self.angRes_in,factor=7)
        self.UpSample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=angRes_in, stride=angRes_in, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels * angRes_out * angRes_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes_out),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=angRes_out, padding=angRes_out, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        xcub= rearrange(x,"b 1 (u h) (v w) -> b (u v) 1 h w ", u=self.angRes_in, v=self.angRes_in)     
        x = rearrange(x,"b 1 (u h) (v w) -> b h w u v", u=self.angRes_in, v=self.angRes_in)
        im11 = x[:, :, :, 0:1,0:1]
        im17 = x[:, :, :, 0:1,1:2]
        im71 = x[:, :, :, 1:2,0:1]
        im77 = x[:, :, :, 1:2,1:2]
        init_FE=self.init_fe(x,im11,im17,im71,im77)
        x_final = self.DeepConvNet(init_FE)
        x_upsample = self.UpSample(x_final)
        xcubic=self.cubic(xcub)
        xcubic=  rearrange(xcubic,"b (u v) 1 h w -> b 1 (h u) (w v)", u=self.angRes_out, v=self.angRes_out)
        x_upsample=x_upsample+xcubic
        replace = rearrange(x_upsample,"b 1 (h u) (w v) -> b h w u v", u=self.angRes_out, v=self.angRes_out)        
        out_replace = torch.cat(( torch.cat((im11, replace[:,:,:,0:1,1:6] , im17), 4), replace[:,:,:,1:6,:], torch.cat((im71, replace[:,:,:,6:7,1:6] , im77), 4)), 3)
        out = rearrange(out_replace,"b h w u v -> b 1 (u h) (v w)")
        return out
        
class CascadedDeepConvGroup(nn.Module):
    def __init__(self, n_group,n_block, angRes_in, channels):
        super(CascadedDeepConvGroup, self).__init__()
        self.n_group = n_group
        self.n_block=n_block
        self.angRes_in = angRes_in
        Groups = []
        for i in range(n_group):
            Groups.append(DeepConvGroup(n_block,angRes_in, channels))
        self.group = nn.Sequential(*Groups)        
        self.ChannelAttention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d((n_group+1) * channels//2, (n_group+1) * channels//(2*8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d((n_group+1) * channels//(2*8), (n_group+1) * channels//2, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d((n_group+1) * channels//2, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        temp = []
        temp.append(x)
        for i in range(self.n_group):
            x = self.group[i](x)
            temp.append(x)
        out_concat = torch.cat(temp, dim=1)
        out_rearrange = rearrange(out_concat,"b c h w u v -> b c (h u) (w v)", u=self.angRes_in, v=self.angRes_in)
        out_CA = self.ChannelAttention(out_rearrange) * out_rearrange        
        return self.fuse(out_CA)

class DeepConvGroup(nn.Module):
    def __init__(self, n_b, angRes_in, channels):
        super(DeepConvGroup, self).__init__()
        self.n_b = n_b
        self.angRes_in = angRes_in
        in_Blocks = []
        for i in range(n_b):
            in_Blocks.append(DeepResBlock(angRes_in, channels))
        self.in_Block = nn.Sequential(*in_Blocks)
        self.ChannelAttention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d((n_b+1) * channels//2, (n_b+1) * channels//(2*8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d((n_b+1) * channels//(2*8), (n_b+1) * channels//2, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d((n_b+1) * channels//2, channels//2, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        temp = []
        #print("input",x.shape)
        temp.append(x)
        for i in range(self.n_b):
            x = self.in_Block[i](x)
            temp.append(x)
        out_concat = torch.cat(temp, dim=1)
        out_rearrange = rearrange(out_concat,"b c h w u v -> b c (h u) (w v)", u=self.angRes_in, v=self.angRes_in)
        out_CA = self.ChannelAttention(out_rearrange) * out_rearrange
        out=self.fuse(out_CA)
        out= rearrange(out,"b c (h u) (w v) -> b c h w u v", u=self.angRes_in, v=self.angRes_in)
        return out
       
class DeepResBlock(nn.Module):
    def __init__(self, angRes_in, channels):
        super(DeepResBlock, self).__init__()
        self.angRes_in = angRes_in 
        self.RB = nn.Sequential(
            nn.Conv2d(channels//2, channels, kernel_size=3, stride=1,  dilation=self.angRes_in, padding=self.angRes_in, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1,  dilation=self.angRes_in, padding=self.angRes_in, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1,  dilation=self.angRes_in, padding=self.angRes_in, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1,  dilation=self.angRes_in, padding=self.angRes_in, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=self.angRes_in, padding=self.angRes_in, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1,  dilation=self.angRes_in, padding=self.angRes_in, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1,  dilation=self.angRes_in, padding=self.angRes_in, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels//2, kernel_size=3, stride=1, dilation=self.angRes_in, padding=self.angRes_in, bias=False),
        )  
        self.conv_1 = nn.Conv2d(channels//2, channels//2, kernel_size=3,dilation=self.angRes_in, padding=self.angRes_in, bias=False)
        self.conv_2 = nn.Conv2d(channels//2, channels//2, kernel_size=3,dilation=self.angRes_in, padding=self.angRes_in, bias=False)

    def forward(self, x):
        x=rearrange(x,"b c h w u v -> b c (h u) (w v)", u=self.angRes_in, v=self.angRes_in)
        x_CONV1 = self.conv_1(x)
        x_out = self.conv_2( self.RB(x_CONV1)+ x_CONV1 )
        x_out = rearrange(x_out,"b c (h u) (w v) -> b c h w u v", u=self.angRes_in, v=self.angRes_in)
        return x_out
class Interpolation(nn.Module):
    def __init__(self, angular_in, factor):
        super(Interpolation, self).__init__()
        self.an = angular_in
        self.an_out = factor#angular_in*factor
        self.factor = factor
    def forward(self, x_mv):
        b, n, c, h, w = x_mv.shape
        x = x_mv.contiguous().view(b, n, c, h*w)
        x = torch.transpose(x, 1, 3)
        x = x.contiguous().view(b*h*w, c, self.an, self.an)
        out = functional.interpolate(x, size=(self.factor, self.factor), mode='bicubic', align_corners=False)#scale_factor=self.factor, mode='bicubic', align_corners=False)
        out = out.view(b,h*w,c,self.an_out*self.an_out)
        out = torch.transpose(out,1,3)
        out = out.contiguous().view(b, self.an_out*self.an_out, c, h, w)   #[N*81,c,h,w]
        return out

class init_FE_Block(nn.Module):
    def __init__(self, angRes_in, channels):
        super(init_FE_Block, self).__init__()
        self.angRes_in = angRes_in 
        self.HV_init_conv = nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.D_init_conv = nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.HV_RB = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.D_RB = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
    def forward(self, x,im11,im17,im71,im77):

        x_h = rearrange(x,"b h w u v -> (b u) v w h")
        x_v = rearrange(x,"b h w u v -> (b v) u h w")
        x_h_FE = self.HV_init_conv(x_h)
        x_v_FE = self.HV_init_conv(x_v)
        x_h_FE = rearrange(x_h_FE,"(b u) (v c) w h -> b c h w u v", u=self.angRes_in, v=self.angRes_in)
        x_v_FE = rearrange(x_v_FE,"(b v) (u c) h w -> b c h w u v", u=self.angRes_in, v=self.angRes_in)
        
        #diagonal
        right_diagonal=torch.cat((im11.unsqueeze(1).squeeze(4,5),im77.unsqueeze(1).squeeze(4,5)),1)
        left_diagonal=torch.flip(torch.cat((im71.unsqueeze(1).squeeze(4,5),im17.unsqueeze(1).squeeze(4,5)),1),[2])
        right_diagonal_FE=self.D_init_conv(right_diagonal)
        rt,rb=torch.chunk(right_diagonal_FE,2,dim=1)
        left_diagonal_FE=torch.flip(self.D_init_conv(left_diagonal),[2])
        lb,lt=torch.chunk(left_diagonal_FE,2,dim=1)
        x_D_FE=torch.cat(( torch.cat((rt , lt),3), torch.cat((lb, rb), 3)), 2)
        x_D_FE = rearrange(x_D_FE,"b c (u h) (v w) -> b c h w u v", u=self.angRes_in, v=self.angRes_in)
        
        x_FE = x_h_FE + x_v_FE+x_D_FE
        fe11 = x_FE[:, :, :, :, 0:1,0:1]
        fe17 = x_FE[:, :, :, :, 0:1,1:2]
        fe71 = x_FE[:, :, :, :, 1:2,0:1]
        fe77 = x_FE[:, :, :, :, 1:2,1:2]
        
        #diagonal
        right_diagonal_fe=torch.cat((fe11.squeeze(4,5),fe77.squeeze(4,5)),1)
        left_diagonal_fe=torch.flip(torch.cat((fe71.squeeze(4,5),fe17.squeeze(4,5)),1),[2])
        right_D_FE=self.D_RB(right_diagonal_fe)
        rt,rb=torch.chunk(right_D_FE,2,dim=1)
        left_D_FE=torch.flip(self.D_RB(left_diagonal_fe),[2])
        lb,lt=torch.chunk(left_D_FE,2,dim=1)
        x_D_FE=torch.cat(( torch.cat((rt , lt),3), torch.cat((lb, rb), 3)), 2)
        x_D_FE = rearrange(x_D_FE,"b c (u h) (v w) -> b c h w u v", u=self.angRes_in, v=self.angRes_in)
        
        x_h = rearrange(x_FE,"b c h w u v -> (b u) (v c) w h")
        x_v = rearrange(x_FE,"b c h w u v -> (b v) (u c) h w")
        x_h_RB = self.HV_RB(x_h)
        x_v_RB = self.HV_RB(x_v)
        x_h_FE = rearrange(x_h_RB,"(b u) (v c) w h -> b c h w u v", u=self.angRes_in, v=self.angRes_in)
        x_v_FE = rearrange(x_v_RB,"(b v) (u c) h w -> b c h w u v", u=self.angRes_in, v=self.angRes_in)
       
        x_out = x_h_FE + x_v_FE +x_D_FE +x_FE
        return x_out

if __name__ == "__main__":
    net = Net(angRes_in=2, angRes_out=7).cuda()
    from thop import profile
    input = torch.randn(1, 1, 192, 192).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops * 2 / 1e9))	
