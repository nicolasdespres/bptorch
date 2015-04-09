--[[

3D tensor applied 

\sum a_i b_i c_i

\sum (a_i . x) (b_i . y) c_i


]]

local MOperator, parent = torch.class('nn.MOperator', 'nn.Module')

function MOperator:__init(inputSizeX, inputSizeY, outputSize, nOperators) 
    self.weight_a = torch.Tensor(inputSizeX, nOperators)
    self.weight_b = torch.Tensor(inputSizeY, nOperators)
    self.weight_c = torch.Tensor(nOperators, outputSize) 

    self.gradWeight_a = torch.Tensor(inputSizeX, nOperators)
    self.gradWeight_a = torch.Tensor(inputSizeY, nOperators)
    self.gradWeight_a = torch.Tensor(nOperators, outputSize)

    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()

    self:reset()
end

function MOperator:reset(stdv)
   function init(w)
       if stdv then
          stdv = stdv * math.sqrt(3)
       else
          stdv = 1. / math.sqrt(w:size(1))
       end
       w:uniform(-stdv, stdv)
   end
   init(self.weight_a)
   init(self.weight_b)
   init(self.weight_c)
end


function MOperator:updateOutput(table)
   local input_x, input_y = unpack(table)

   if (input_x:dim() ~= 1 and input_x:dim() ~= 2) or input_y:dim() ~= 2 then
      error('inputs must be matrices')
   end

   self.output:resize(input_y:size(1), self.weight_c:size(2))

   -- v is a (batch_size x nOperators matrix)
   local v = input_y * self.weight_b

   if input_x:dim() == 1 then
       local y = self.weight_a:t() * input_x
       y = y:view(1, -1)
       y = y:expand(input_y:size(1), y:size(2))
       v:cmul(y)
   else 
       v:cmul(input_x * self.weight_a)
   end
   
   self.output:mm(v, self.weight_c)
   return self.output
end

function MOperator:updateGradInput(table, gradOutput)
   if self.gradInput then
      local input, targets = unpack(table)



      return self.gradInput
   end
end

function MOperator:accGradParameters(table, gradOutput, scale)
   local input, targets = unpack(table)
   scale = scale or 1
end