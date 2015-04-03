--[[

Hierarchical log-softmax with a tree structure

x is the input
L(c) is the path from the 

P(class is c | x) = \prod_

--]]

local HSoftMax, parent = torch.class('nn.HSoftMax', 'nn.Module')

-- inputSize: the size of the incoming vector
-- hierarchy: an array containing parents (0 == root), neg = left branch, pos = right branch.
-- leaves come first, followed by the inner nodes
-- last node is the root
function HSoftMax:__init(inputSize, parents)
   parent.__init(self)

   -- @TODO: verify the hierarchy is OK
   self.parents = parents:long()
   local nleaves = (parents:size(1) + 1) / 2
   self.nleaves = nleaves
 
   local innerNodes = parents:size(1) - nleaves

   assert(parents:sum() == 0)
   assert(parents[parents:size(1)] == 0)

   -- Computes the depth at each inner node
   self.depth = torch.IntTensor(innerNodes)
   function compute_depth(ix)
      if self.depth[ix] == -1 then
         local pix = torch.abs(self.parents[ix + nleaves])
         if pix == 0 then
            self.depth[ix] = 0
         else
            self.depth[ix] = compute_depth(pix - nleaves) + 1
         end
      end
      return self.depth[ix]
   end
   self.depth:fill(-1)
   for ix = 1, innerNodes do
      compute_depth(ix)
   end
   
   self.weight = torch.Tensor(innerNodes, inputSize)
   self.bias = torch.Tensor(innerNodes)

   self.gradWeight = torch.Tensor(innerNodes, inputSize)
   self.gradBias = torch.Tensor(innerNodes)
   
   -- No gradient for the targets
   self.gradInput = { torch.Tensor(), nil }

   -- Stores updates
   self.updates = { offsets=torch.LongTensor(), nodes=torch.LongTensor(), values=torch.Tensor()}

   self:reset()
end

function HSoftMax:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1. / math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end
   

function HSoftMax:read(stream, versionNumber)
   local var = stream:readObject()
   for k,v in pairs(var) do
      self[k] = v
   end
   self.updates = { offsets=torch.LongTensor(), nodes=torch.LongTensor(), values=torch.Tensor()}
end

function HSoftMax:write(stream)
   local var = {}
   for k,v in pairs(self) do
      -- Filter fields 
      if k ~= "updates" then
         if torch.File.isWritableObject(v) then
            var[k] = v
         else
            print(string.format('$ Warning: cannot write object field <%s>', k))
         end
      end
   end
   stream:writeObject(var)
end


function HSoftMax:updateOutput(table)
   local input, targets = unpack(table)
   local nframe, output

   if input:dim() ~= 1 and input:dim() ~= 2 then
      error('input must be vector or matrix')
   end

   if input:dim() == 1 then
      nframe = 1
      input = input:view(1, input:size(1))
      self.output:resize(1)
      output = self.output:view(1, 1)
   else
      nframe = input:size(1)
      self.output:resize(nframe, 1)
      output = self.output
   end
   
   input.nn.HSoftMax_updateOutputWithTarget(self, input, targets, output)
   return self.output
end

function HSoftMax:updateGradInput(table, gradOutput)
   local input, targets = unpack(table)

   if self.gradInput then
      local gradInput0 = self.gradInput[1]
      gradInput0:resizeAs(input)
      gradInput0:zero()
   
      -- Adapt to the batch size
      if input:dim() == 1 then
         nframe = 1
         input = input:view(1, input:size(1))
      else
         nframe = input:size(1)
         gradOutput = gradOutput:view(-1)
      end

      input.nn.HSoftMax_updateGradInput(self, gradInput0, gradOutput)

      return self.gradInput
   end
end



function HSoftMax:accGradParameters(table, gradOutput, scale)
   local input, targets = unpack(table)

   scale = scale or 1

   if input:dim() ~= 1 and input:dim() ~= 2 then
      error('input must be vector or matrix')
   end

   if input:dim() == 1 then
      nframe = 1
      input = input:view(1, input:size(1))
   else
      nframe = input:size(1)
      gradOutput = gradOutput:view(-1)
   end

   input.nn.HSoftMax_accGradParameters(self, scale, input, gradOutput)

end

--- OLD CODE ---

if NOOPTIM then

   function HSoftMax:updateOutput(table)
      local input, targets = unpack(table)
      local nframe, output

      if input:dim() ~= 1 and input:dim() ~= 2 then
         error('input must be vector or matrix')
      end

      if input:dim() == 1 then
         nframe = 1
         input = input:view(1, input:size(1))
         self.output:resize(1)
         output = self.output:view(1, 1)
      else
         nframe = input:size(1)
         self.output:resize(nframe, 1)
         output = self.output
      end
      
      -- Stores the heavy computation before computing gradInput or updating gradient
      local tdepth = 0
      for frame = 1, nframe do
         local current = self.parents[targets[frame]]
         local ix = torch.abs(current) - self.nleaves
         tdepth = tdepth + self.depth[ix] + 1
      end
      self.updates.offsets:resize(nframe + 1)

      -- --TODO-- : Compute maximum or average depth
      self.updates.nodes:resize(tdepth)
      self.updates.values:resize(tdepth)
      local offset = 1

      for frame = 1, nframe do
         self.updates.offsets[frame] = offset

         local current = self.parents[targets[frame]]
         local logp = 0
         while current ~= 0 do
            sign = 1.
            if current <= 0 then
               sign = -1.
               current = -current
            end

            local ix = current - self.nleaves
            local current_p = 1 + torch.exp(sign * (self.weight[ix]:dot(input[frame]) + self.bias[ix]))

            self.updates.nodes[offset] = current
            self.updates.values[offset] = sign * (1. / current_p - 1.)

            logp = logp - torch.log(current_p)
            current = self.parents[current]
            offset = offset + 1

         end

         -- Set the output
         output[{frame, 1}] = logp
      end

      -- Marks the end
      self.updates.offsets[nframe+1] = offset
      return self.output
   end

   function HSoftMax:updateGradInput(table, gradOutput)
      local input, targets = unpack(table)

      if self.gradInput then
         local gradInput0 = self.gradInput[1]
         gradInput0:resizeAs(input)
         gradInput0:zero()
      
         -- Adapt to the batch size
         if input:dim() == 1 then
            nframe = 1
            input = input:view(1, input:size(1))
         else
            nframe = input:size(1)
            gradOutput = gradOutput:view(-1)
         end

         for frame = 1, (self.updates.offsets:size(1) - 1) do
            for j = self.updates.offsets[frame], self.updates.offsets[frame+1] - 1 do
               local current = self.updates.nodes[j]
               local ix = current - self.nleaves
               gradInput0[frame]:add(gradOutput[frame] * self.updates.values[j], self.weight[ix])
            end
         end

         return self.gradInput
      end
   end



   function HSoftMax:accGradParameters(table, gradOutput, scale)
      local input, targets = unpack(table)

      scale = scale or 1

      if input:dim() ~= 1 and input:dim() ~= 2 then
         error('input must be vector or matrix')
      end

      if input:dim() == 1 then
         nframe = 1
         input = input:view(1, input:size(1))
      else
         nframe = input:size(1)
         gradOutput = gradOutput:view(-1)
      end

      for frame = 1, self.updates.offsets:size(1) - 1 do
         for j = self.updates.offsets[frame], self.updates.offsets[frame+1] - 1 do
            local current = self.updates.nodes[j]
            local ix = current - self.nleaves
            local c = scale * self.updates.values[j] * gradOutput[frame]
            self.gradWeight[ix]:add(c, input[frame])
            self.gradBias[{{ix}}]:add(c)
         end
      end
   end

end -- NOOPTIM


-- we do not need to accumulate parameters when sharing
HSoftMax.sharedAccUpdateGradParameters = HSoftMax.accUpdateGradParameters
