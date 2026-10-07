// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include "../include/potential.h"
#include <iostream>
#include "../include/parameters.h"
#include "../include/define.h"

// GPU copies of the pair potentials and field coupling of potential.cpp (see the comments there); keep identical.

__device__ float Bulk_Energy_Selinger_Pear_GPU(float ni[3], float nj[3], Parameters *params, float rij[3]){

  float nij, polar_splay;
      nij=ni[0]*nj[0]+ni[1]*nj[1]+ni[2]*nj[2];
      polar_splay=+(1+2*nij+nij*nij)*((nj[0]-ni[0])*rij[0]
      +(nj[1]-ni[1])*rij[1]+(nj[2]-ni[2])*rij[2]);
  return (-params->A*nij*nij -params->B1*nij - (params->C*0.25)*polar_splay);
}

__device__ float Bulk_Energy_Lebwohl_Lasher_GPU(float ni[3], float nj[3], Parameters *params, float rij[3]){

  float nij=ni[0]*nj[0]+ni[1]*nj[1]+ni[2]*nj[2];
  return -params->A*1.5*nij*nij;
}

__device__ float Bulk_Energy_GHRL_GPU(float ni[3], float nj[3], Parameters *params, float rij[3]){
  
  const float el=params->ghrl_lambda;
  const float em=params->ghrl_mu;
  const float en=params->ghrl_nu;
  const float er=params->ghrl_rho;
  const float es=params->ghrl_sigma;
  float ai=ni[0]*rij[0]+ni[1]*rij[1]+ni[2]*rij[2];
  float aj=nj[0]*rij[0]+nj[1]*rij[1]+nj[2]*rij[2];
  float nij=ni[0]*nj[0]+ni[1]*nj[1]+ni[2]*nj[2];
  float pij=((float)1.5)*nij*nij-((float)0.5);
  float cross=(ni[2]*nj[1]-ni[1]*nj[2])*rij[0]
             +(ni[0]*nj[2]-ni[2]*nj[0])*rij[1]
             +(ni[1]*nj[0]-ni[0]*nj[1])*rij[2];
  
  float E1 =((((float)1.5)*ai*ai)+(((float)1.5)*aj*aj)-1);
  return ((E1*(er*pij+el)+em*(ai*aj*nij-(float)1.0/9))+ en*pij+es*(nij>0?1:-1)*cross);
}
__device__ float Electric_Potential_GPU(float ni[3], Parameters *params){
  float nDotE = params->elecE*(ni[0]*params->elecX+ni[1]*params->elecY+ni[2]*params->elecZ);

  return -params->elecP*nDotE - params->elecA*nDotE*nDotE;
}

__device__ int Periodic_Boundary_GPU(int &ii, int NN){
   ii=((ii+NN)%NN);
   return 1;
}

__device__ int Free_Boundary_GPU(int &ii, int NN){
   if (ii==-1||ii==NN) return 0;
   else                 return 1 ;
}
