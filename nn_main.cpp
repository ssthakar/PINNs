#include "nn_main.h"
#include "utils.h"
#include <ATen/TensorIndexing.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/flatten.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/meshgrid.h>
#include <ATen/ops/mse_loss.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros_like.h>
#include <c10/core/DispatchKeySet.h>
#include <cassert>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/linear.h>
#include <torch/serialize/input-archive.h>
#include <torch/serialize/output-archive.h>
#include <vector>
#include <cmath>
//- loads in python like indaexing of tensors
using namespace torch::indexing; 

//-------------------PINN definitions----------------------------------------//

//- function to create layers present in the net
void PinNetImpl::create_layers()
{
  //- register input layer 
  input = register_module
  (
    "fc_input",
    torch::nn::Linear(INPUT_DIM,HIDDEN_LAYER_DIM)
  );
  
  //- register and  hidden layers 
  for(int i=0;i<N_HIDDEN_LAYERS;i++)
  {
    //- hiden layer name
    std::string layer_name = "fc_hidden" + std::to_string(i);
    
    //- register each hidden layer
    torch::nn::Linear linear_layer = register_module
    (
      layer_name,
      torch::nn::Linear(HIDDEN_LAYER_DIM,HIDDEN_LAYER_DIM)
    );
     
    //- intialize network parameters
    torch::nn::init::xavier_normal_(linear_layer->weight);
    
    // std::cout<<"weights for layer "<<i<<linear_layer->weight<<std::endl;
    
    //- populate sequential with layers
    hidden_layers->push_back(linear_layer);

    //- register activation functions 
    hidden_layers->push_back
    (
      register_module
      (
        "fc_relu_hidden" + std::to_string(i), 
        torch::nn::Tanh()
      )
    );
  }
  
  //- register output layer
  output = register_module
  (
    "fc_output",
    torch::nn::Linear(HIDDEN_LAYER_DIM,OUTPUT_DIM)
  );
}

//- constructor for PinNet module implementation
PinNetImpl::PinNetImpl
(
  const Dictionary &netDict
)
: 
  dict(netDict), //pass in Dictionary 
  INPUT_DIM(dict.get<int>("inputDim")), // no. of input features  
  HIDDEN_LAYER_DIM(dict.get<int>("hiddenLayerDim")), // no. of neurons in HL
  N_HIDDEN_LAYERS(dict.get<int>("nHiddenLayer")), // no. of hidden layers
  OUTPUT_DIM(dict.get<int>("outputDim")) //- no. of output features
{
  //- set parameters from Dictionary lookup
  N_EQN = dict.get<int>("NEQN");
  N_BC = dict.get<int>("NBC");
  N_IC = dict.get<int>("NIC");
  //- boolean for transient or steady state mode
  transient_ = dict.get<int>("transient");
  //- get target loss from dict
  ABS_TOL = dict.get<float>("ABSTOL");
  //- patch size for pde loss
  BATCHSIZE=dict.get<int>("BATCHSIZE");
  //- creat the layers in the net
  create_layers();
}


//- forward propagation 
torch::Tensor PinNetImpl::forward
(
 const torch::Tensor& X
)
{
  torch::Tensor I = torch::tanh(input(X));
  I = hidden_layers->forward(I);
  I = output(I);
  return I;
}
  

//------------------end PINN definitions-------------------------------------//


//-----------------derivative definitions------------------------------------//

//- first order derivative
torch::Tensor d_d1
(
  const torch::Tensor &I,
  const torch::Tensor &X,
  int spatialIndex
)
{
  torch::Tensor derivative = torch::autograd::grad 
  (
    {I},
    {X},
    {torch::ones_like(I)},
    true,
    true,
    true
  )[0].requires_grad_(true);
  return derivative.index({Slice(),spatialIndex});
}

torch::Tensor d_d1
(
  const torch::Tensor &I,
  const torch::Tensor &X
)
{
  torch::Tensor derivative = torch::autograd::grad 
  (
    {I},
    {X},
    {torch::ones_like(I)},
    true,
    true,
    true
  )[0].requires_grad_(true);
  return derivative;
}

//- higher order derivative
torch::Tensor d_dn
(
  const torch::Tensor &I, 
  const torch::Tensor &X, 
  int order, // order of derivative
  int spatialIndex
)
{
  torch::Tensor derivative =  d_d1(I,X,spatialIndex);
  for(int i=0;i<order-1;i++)
  {
    derivative = d_d1(derivative,X,spatialIndex);
  }
  return derivative;
}

//- function overload when X is 1D tensor
torch::Tensor d_dn
(
  const torch::Tensor &I, 
  const torch::Tensor &X, 
  int order // order of derivative
)
{
  torch::Tensor derivative =  d_d1(I,X);
  for(int i=0;i<order-1;i++)
  {
    derivative = d_d1(derivative,X);
  }
  return derivative;
}

//----------------------------end derivative definitions---------------------//


//----------------------------CahnHillard function definitions---------------//

//- I is the input tensor  having the shape {u,v,p,C} X N_BATCHSIZE

//- thermoPhysical properties for mixture
torch::Tensor CahnHillard::thermoProp
(
  float propLiquid, //thermoPhysical prop of liquid  phase 
  float propGas, // thermoPhysical prop of gas phase
  const torch::Tensor &I
)
{
  //- get phase field var 
  const torch::Tensor &C = I.index({Slice(),3});
  torch::Tensor mixtureProp = 
    0.5*(1+C)*propLiquid + 0.5*(1-C)*propGas;
  return mixtureProp;
}

//- continuity loss 
torch::Tensor CahnHillard::L_Mass2D
(
  const mesh2D &mesh 
)
{
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  torch::Tensor du_dx = d_d1(u,mesh.iPDE_,0);
  torch::Tensor dv_dy = d_d1(u,mesh.iPDE_,1);
  torch::Tensor loss = du_dx+dv_dy;
  return torch::mse_loss(loss, torch::zeros_like(loss));
}

//- returns the phi term needed 
torch::Tensor CahnHillard::phi
(
  const mesh2D &mesh
)
{
  float &e = mesh.thermo_.epsilon;
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  torch::Tensor Cxx = d_dn(C,mesh.iPDE_,2,0);
  torch::Tensor Cyy = d_dn(C,mesh.iPDE_,2,1);
  return C*(C*C-1) - e*e*(Cxx + Cyy); 
}

//- returns CahnHillard Loss
torch::Tensor CahnHillard::CahnHillard2D
(
  const mesh2D &mesh
)
{
  const float &e = mesh.thermo_.epsilon;
  const float &Mo = mesh.thermo_.Mo;
  //- u vel
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  //- v vel
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  //- phase field var
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  //- derivatives 
  torch::Tensor dC_dt = d_d1(C,mesh.iPDE_,2);
  torch::Tensor dC_dx = d_d1(C,mesh.iPDE_,0);
  torch::Tensor dC_dy = d_d1(C,mesh.iPDE_,1);
  torch::Tensor phi = CahnHillard::phi(mesh);
  torch::Tensor dphi_dxx = d_dn(phi,mesh.iPDE_,2,0);
  torch::Tensor dphi_dyy = d_dn(phi,mesh.iPDE_,2,1);
  //- loss term
  torch::Tensor loss = dC_dt + u*dC_dx + v*dC_dy - 
    Mo*(dphi_dyy + dphi_dyy);
  return torch::mse_loss(loss,torch::zeros_like(loss));
}

//- returns the surface tension tensor needed in mom equation
torch::Tensor CahnHillard::surfaceTension
(
  const mesh2D &mesh,
  int dim
)
{
  const float &e_inv = 1.0/mesh.thermo_.epsilon;
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  torch::Tensor surf = e_inv*mesh.thermo_.C*CahnHillard::phi(mesh)
    *d_d1(C,mesh.iPDE_,dim);
  return surf;
} 

//- momentum loss for x direction in 2D 
torch::Tensor CahnHillard::L_MomX2d
(
  const mesh2D &mesh
)
{ 
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  const torch::Tensor &p = mesh.fieldsPDE_.index({Slice(),2});
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
}

//- momentum loss for y direction in 2D
torch::Tensor CahnHillard::L_MomY2d
(
  const mesh2D &mesh
)
{
}

//- get total PDE loss
torch::Tensor CahnHillard::PDEloss(mesh2D &mesh)
{
  //- loss from mass conservation
  torch::Tensor LM = CahnHillard::L_Mass2D(mesh);
  torch::Tensor LMX = CahnHillard::L_MomX2d(mesh);
  torch::Tensor LMY = CahnHillard::L_MomX2d(mesh);
  torch::Tensor LC = CahnHillard::CahnHillard2D(mesh);
  //- return total pde loss
  return LM + LC + LMX + LMY;
}

torch::Tensor CahnHillard::slipWall(torch::Tensor &I, torch::Tensor &X,int dim)
{
  const torch::Tensor &u = I.index({Slice(),0});  
  const torch::Tensor &v = I.index({Slice(),1});
  torch::Tensor dv_dx = d_d1(v,X,dim);
  return torch::mse_loss(dv_dx,torch::zeros_like(dv_dx))
    + torch::mse_loss(u,torch::zeros_like(u));
}

torch::Tensor CahnHillard::noSlipWall(torch::Tensor &I, torch::Tensor &X)
{
  const torch::Tensor &u = I.index({Slice(),0});
  const torch::Tensor &v = I.index({Slice(),1});
  return torch::mse_loss(u,torch::zeros_like(u))  + 
    torch::mse_loss(v,torch::zeros_like(v));
  
}
//- get boundary loss
torch::Tensor CahnHillard::BCloss(mesh2D &mesh)
{
  torch::Tensor lossLeft = CahnHillard::slipWall(mesh.fieldsLeft_, mesh.iLeftWall_,0);
  torch::Tensor lossRight = CahnHillard::slipWall(mesh.fieldsRight_,mesh.iRightWall_, 0);
  torch::Tensor lossTop = CahnHillard::noSlipWall(mesh.fieldsTop_, mesh.iTopWall_);
  torch::Tensor lossBottom = CahnHillard::noSlipWall(mesh.fieldsBottom_, mesh.iBottomWall_);
  return lossLeft + lossRight + lossTop + lossBottom;
}

//- get the intial loss for the 
torch::Tensor CahnHillard::ICloss(mesh2D &mesh)
{
  //- x vel
  const torch::Tensor &u = mesh.fieldsIC_.index({Slice(),0});
  //- y vel
  const torch::Tensor &v = mesh.fieldsIC_.index({Slice(),1});
  //- phaseField variable
  const torch::Tensor &C = mesh.fieldsIC_.index({Slice(),2});
  
  torch::Tensor uLoss = torch::mse_loss(u, torch::zeros_like(u));
  torch::Tensor vLoss = torch::mse_loss(v, torch::zeros_like(u));
  torch::Tensor CLoss = torch::mse_loss(C,CahnHillard::C_at_IntialTime(mesh));
  return uLoss +vLoss +CLoss;
}


torch::Tensor CahnHillard::loss(mesh2D &mesh)
{
  // torch::Tensor pdeloss = CahnHillard::PDEloss(mesh);
  torch::Tensor bcLoss = CahnHillard::BCloss(mesh);
  torch::Tensor pdeLoss = CahnHillard::PDEloss(mesh);
  torch::Tensor icLoss = CahnHillard::ICloss(mesh);
  return bcLoss + pdeLoss + icLoss; //+ bcloss;
}

torch::Tensor CahnHillard::C_at_IntialTime(mesh2D &mesh)
{
  if(mesh.lbT_ == 0)
  {
    const float &xc = mesh.xc;
    const float &yc = mesh.yc;
    const float &e = mesh.thermo_.epsilon;
    //- x 
    const torch::Tensor &x = mesh.iIC_.index({Slice(),0});
    //- y
    const torch::Tensor &y = mesh.iIC_.index({Slice(),1});
    torch::Tensor Ci =torch::tanh((torch::sqrt(torch::pow(x - xc, 2) + torch::pow(y - yc, 2)) - 0.15)/ (1.41421356237 * e));
    
    return Ci;
  }
  else 
  {
    torch::Tensor Ci = mesh.netPrev_->forward(mesh.iIC_).index({Slice(),3});
    return Ci;
  }
}


//---------------------end CahnHillard function definitions------------------//

torch::Tensor Heat::L_Diffusion2D
(
  mesh2D &mesh
)
{
  float PI = 3.14159265358979323846;
  torch::Tensor u_xx = d_dn(mesh.fieldsPDE_,mesh.iPDE_,2,0);
  torch::Tensor u_yy = d_dn(mesh.fieldsPDE_,mesh.iPDE_,2,1);
  torch::Tensor fTerm =
    -2*sin(PI*mesh.iPDE_.index({Slice(),0}))*sin(PI*mesh.iPDE_.index({Slice(),1}))*PI*PI;
  return torch::mse_loss(u_xx+u_yy,fTerm);

}


//- total loss for 2d diffusion equation
torch::Tensor Heat::loss(mesh2D &mesh)
{
  //- create samples
  // mesh.createTotalSamples(net,2,mesh.xyGrid);
  torch::Tensor pdeloss = Heat::L_Diffusion2D(mesh);
  torch::Tensor l = torch::mse_loss(mesh.fieldsLeft_,torch::zeros_like(mesh.fieldsLeft_));
  
  torch::Tensor r = torch::mse_loss(mesh.fieldsRight_,torch::zeros_like(mesh.fieldsRight_));

  torch::Tensor t = torch::mse_loss(mesh.fieldsTop_,torch::zeros_like(mesh.fieldsTop_));

  torch::Tensor b = torch::mse_loss(mesh.fieldsBottom_,torch::zeros_like(mesh.fieldsBottom_));

  return l+b+r+t+pdeloss;
  
}
//---------------------------mesh2d function definitions---------------------//

  

//- construct computational domain for the PINN instance
mesh2D::mesh2D
(
  Dictionary &meshDict, //mesh parameters
  PinNet &net,
  PinNet &netPrev,
  torch::Device &device, // device info
  thermoPhysical &thermo
):
  net_(net), // pass in current neural net
  netPrev_(netPrev), // pass in other neural net
  dict(meshDict),
  device_(device), // pass in device info
  thermo_(thermo), // pass in thermo class instance
  lbX_(dict.get<float>("lbX")), // read in mesh props from dict
  ubX_(dict.get<float>("ubX")),
  lbY_(dict.get<float>("lbY")),
  ubY_(dict.get<float>("ubY")),
  lbT_(dict.get<float>("lbT")),
  ubT_(dict.get<float>("ubT")),
  deltaX_(dict.get<float>("dx")),
  deltaY_(dict.get<float>("dy")),
  deltaT_(dict.get<float>("dt")),
  xc(dict.get<float>("xc")),
  yc(dict.get<float>("yc"))

{
  TimeStep_ = dict.get<float>("stepSize");
    //- get number of ponits from bounds and step size
  Nx_ = (ubX_ - lbX_)/deltaX_ + 1;
  Ny_ = (ubY_ - lbY_)/deltaY_ + 1;
  Nt_ = (ubT_ - lbT_)/deltaT_ + 1;

  //- total number of points in the entire domain (nDOF)
  Ntotal_ = Nx_*Ny_*Nt_;
  
  std::cout<<"Creating dimensional grids: "<<std::endl;

  //- populate the individual 1D grids
  xGrid = torch::linspace(lbX_, ubX_, Nx_,device_);
  yGrid = torch::linspace(lbY_, ubY_, Ny_,device_);
  tGrid = torch::linspace(lbT_, ubT_, Nt_,device_);
  
  std::cout<<"done !\n"<<std::endl;

  std::cout<<"Creating mesh: "<<std::endl;

  //- construct entire mesh domain
  mesh_ = torch::meshgrid({xGrid,yGrid,tGrid});
  
  std::cout<<"done !\n"<<std::endl;
  
  std::cout<<"Creating spatial grid: "<<std::endl;
  xyGrid = torch::meshgrid({xGrid,yGrid});

  xy = torch::stack({xyGrid[0].flatten(),xyGrid[1].flatten()},1);
  xy.set_requires_grad(true);
  std::cout<<"done !\n"<<std::endl;
  
  std::cout<<"Creating Boundary grids: "<<std::endl;

  //- create boundary grids
  createBC();
  /*
  //- create boundary grids for heat and boundary prediction
  il = torch::stack({leftWall[0].flatten(),leftWall[1].flatten()},1);
  il.set_requires_grad(true);  
  ir = torch::stack({rightWall[0].flatten(),rightWall[1].flatten()},1);
  ir.set_requires_grad(true);
  it = torch::stack({topWall[0].flatten(),topWall[1].flatten()},1);
  it.requires_grad_(true);
  ib = torch::stack({bottomWall[0].flatten(),bottomWall[1].flatten()},1);
  ib.set_requires_grad(true);
  */
  std::cout<<"done !\n"<<std::endl;
  
  std::cout<<"Initializing solution fields: "<<std::endl;

  //- initialize the flow field
  initialize();
  
  std::cout<<"done !\n"<<std::endl;
}

//- operator overload () to acess main computational domain
torch::Tensor  mesh2D::operator()(int i, int j, int k)  
{
  return torch::stack
  (
    {
      mesh_[0].index({i, j, k}), 
      mesh_[1].index({i, j, k}), 
      mesh_[2].index({i, j, k})
    }
  ); 
}

//- intialize solution fields
void mesh2D::initialize()
{
  //- if t = 0 
  if(lbT_ == 0)
  {

  }
  else 
  {
    
  }
}

//- TODO: cout info
void mesh2D::createBC()
{
  
  torch::Tensor xLeft = torch::tensor(lbX_,device_);
  torch::Tensor xRight = torch::tensor(ubX_,device_);
  torch::Tensor yBottom = torch::tensor(lbY_, device_);
  torch::Tensor yTop = torch::tensor(ubY_, device_);
  torch::Tensor tInitial = torch::tensor(lbT_,device_);
  if(net_->transient_==1)
  {
    leftWall = torch::meshgrid({xLeft,yGrid,tGrid});
    rightWall = torch::meshgrid({xRight,yGrid,tGrid});
    topWall = torch::meshgrid({xGrid,yTop,tGrid});
    bottomWall = torch::meshgrid({xGrid,yBottom,tGrid});
    initialGrid_ = torch::meshgrid({xGrid,yGrid,tInitial});
  }
  else 
  {
    leftWall = torch::meshgrid({xLeft,yGrid});
    rightWall = torch::meshgrid({xRight,yGrid});
    topWall = torch::meshgrid({xGrid,yTop});
    bottomWall = torch::meshgrid({xGrid,yBottom});
  }
}

void mesh2D::createSamples
(
  std::vector<torch::Tensor> &grid, 
  torch::Tensor &samples,
  int nSamples
) 
{
  //- vectors to stack
  std::vector<torch::Tensor> vectorStack;
  
  //- total number of points in the grid
  int ntotal = grid[0].numel();
  
  //- random indices for PDE loss
  torch::Tensor pdeIndices_ = torch::randperm
  (ntotal,device_).slice(0,0,nSamples);
  
  //- push vectors to vectors stack
  for(int i=0;i<grid.size();i++)
  {
    vectorStack.push_back
    (
      torch::flatten
      (
        grid[i]
      ).index_select(0,pdeIndices_)
    );
  }

  //- pass stack to get samples
  samples = torch::stack(vectorStack,1);
  
  //- set gradient =true
  samples.set_requires_grad(true);
}



//- generate sampling points for IC loss
void mesh2D::getICsamples(int N_IC)
{
  //- random indices for IC loss
  torch::Tensor icIndices_ = 
    torch::randperm(Ntotal_,device_).slice(0,0,N_IC);
  
  //- get sampling points for x,y,t
  torch::Tensor xIC_ = 
    torch::flatten(mesh_[0]).index_select(0,icIndices_);
  torch::Tensor yIC_ = 
    torch::flatten(mesh_[1]).index_select(0,icIndices_);
  
  //- time input is lower bound for all spatial inputs
  iIC_ = torch::stack
  (
    {
      xIC_,
      yIC_,
      torch::full_like(xIC_,lbT_,device_)
    },1
  );
}

//- create the total samples required for neural net
void mesh2D::createTotalSamples
(
  int iter
) 
{
  createIndices();
  if(net_->transient_==0)
  {
    torch::Tensor batchIndices = torch::slice
    (
      pdeIndices_,
      0,
      iter*net_->BATCHSIZE,
      (iter + 1)*net_->BATCHSIZE
    );
    createSamples(xyGrid,iPDE_,batchIndices);
  }
  else
  {

    torch::Tensor batchIndices = pdeIndices_.slice
    (
      0,
      iter*net_->BATCHSIZE,
      (iter + 1)*net_->BATCHSIZE
    );
    createSamples(mesh_,iPDE_,batchIndices);
    
  }
  //- generate samples for initial condition residual
  
  if(net_->transient_ == 1)
  {
    // getICsamples(net_->N_IC);
    createSamples(initialGrid_,iIC_,net_->N_IC);
  }
  //- generate samples for boundary condtorch::nn clonerition residual
  
  //- update samples for left wall 
  createSamples(leftWall,iLeftWall_,net_->N_BC);
  
  //- update samples for right wall 
  createSamples(rightWall, iRightWall_,net_->N_BC);
  
  //- update samples for top wall 
  createSamples(topWall,iTopWall_,net_->N_BC);
  
  //- update samples for bottom wall
  createSamples(bottomWall,iBottomWall_,net_->N_BC); 
}

void mesh2D::update(int iter)
{ 
  createTotalSamples(iter);
  //- update all fields
  fieldsPDE_ = net_->forward(iPDE_);
  if(net_->transient_ == 1)
  { 
    fieldsIC_ = net_->forward(iIC_);
  }
  fieldsLeft_ = net_->forward(iLeftWall_);
  fieldsRight_ = net_->forward(iRightWall_);
  fieldsBottom_ = net_->forward(iBottomWall_);
  fieldsTop_ = net_->forward(iTopWall_);
}

//- creates indices tensor for iPDE
void mesh2D::createIndices()
{
  if(net_->transient_==0)
  {
    pdeIndices_ = 
      torch::randperm(xyGrid[0].numel(),device_).slice(0,0,net_->N_EQN);
  }
  else
  {
    pdeIndices_ = 
      torch::randperm(mesh_[0].numel(),device_).slice(0,0,net_->N_EQN);
  }
}

void mesh2D::createSamples
(
 std::vector<torch::Tensor> &grid,
 torch::Tensor &samples,
 torch::Tensor &indices
)
{
  //- vectors to stack
  std::vector<torch::Tensor> vectorStack;
  //- push vectors to vectors stack
  for(int i=0;i<grid.size();i++)
  {
    vectorStack.push_back
    (
      torch::flatten
      (
        grid[i]
      ).index_select(0,indices)
    );
  }

  //- pass stack to get samples
  samples = torch::stack(vectorStack,1);
  
  //- set gradient =true
  samples.set_requires_grad(true);

}

void mesh2D::updateMesh()
{
  //- update the lower level of time grid
  lbT_ = lbT_ + TimeStep_;
  //- get new number of time steps in the current time domain
  Nt_ = (ubT_ - lbT_)/deltaT_ + 1;
  //- update tGrid
  tGrid = torch::linspace(lbT_, ubT_, Nt_,device_);
  //- update main mesh
  mesh_ = torch::meshgrid({xGrid,yGrid,tGrid});
  
}

//-------------------------end mesh2D definitions----------------------------//

/*
 * Format for thermoPhysical text file 
 * 
*/

//---thermophysical class definition
thermoPhysical::thermoPhysical(Dictionary &dict)
{
  Mo = dict.get<float>("Mo");
  epsilon = dict.get<float>("epsilon");
  sigma0 = dict.get<float>("sigma0");
  muL = dict.get<float>("muL");
  muG = dict.get<float>("muG");
  rhoL = dict.get<float>("rhoL");
  rhoG = dict.get<float>("rhoG");
  C = 1.06066017178;
}

void loadState(PinNet& net1, PinNet &net2)
{
  torch::autograd::GradMode::set_enabled(false);
  auto new_params = net2->named_parameters();
  auto params = net1->named_parameters(true);
  auto buffer = net1->named_buffers(true);
  for(auto &val : new_params)
  {
    auto name = val.key();
    auto *t = params.find(name);
    if(t!=nullptr)
    {
      t->copy_(val.value());
    }
    else
    {
      t= buffer.find(name);
      if (t !=nullptr)
      {
        t->copy_(val.value());
      }
    }
  }
  torch::autograd::GradMode::set_enabled(true);
} 


















