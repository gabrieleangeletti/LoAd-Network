require "nn"

--[[
   Spatial Pyramid Pooling layer.

   Applies a Max Pooling operation at different window sizes and strides,
   and then concatenates together all the outputs. The advantage is that
   the output size is the same regardless of the input size.
   Example:
      VGG16 last convolutional feature maps have shape Nx512x14x14.
      By using a 4-levels Spatial Pyramid Pooling with levels 4x4, 3x3, 2x2, 1x1
      The output will have shape N x 15360, where
      15360 = (4*4*512) + (3*3*512) + (2*2*512) + 512.
      This remains true also in the case of feature maps different from 14x14.
]]--

local SpatialPyramidPooling, Parent = torch.class('nn.SpatialPyramidPooling', 'nn.Concat')

function SpatialPyramidPooling:__init(...)
   --[[
        Parameters
        ----------
        levels : tables
                 Ex: SpatialPyramidPooling({4, 4}, {3, 3}) for a pyramid with two
                 levels: 4x4 and 3x3
   ]]--
    Parent.__init(self, 2)
    local args = {...}
    for k, v in ipairs(args) do
        Parent.add(self, nn.Sequential()
            :add(nn.SpatialAdaptiveMaxPooling(v[1], v[2]))
            :add(nn.View(-1):setNumInputDims(3))
            :add(nn.Contiguous()))
    end
end

function SpatialPyramidPooling:updateOutput(input)
   --[[
         Parameters
         ----------
         input : 3D or 4D Tensor
                 Convolutional feature maps

         Returns
         -------
         3D or 4D Tensor
            Spatial Pyramid Pooling operation output
   ]]--
   return Parent.updateOutput(self, input)
end

function SpatialPyramidPooling:updateGradInput(input, gradOutput)
   --[[
         Parameters
         ----------

         input : 2D fully-connected features (num_samples x num_features)

     gradOutput : gradient wrt to module's output.
   ]]--
    return Parent.updateGradInput(self, input, gradOutput)
end
