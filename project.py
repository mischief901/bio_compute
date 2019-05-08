import math
import operator
import numpy as np
import edebug as edb
from numpy import random
from copy import deepcopy
import sim_toolbox as stb
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


##testing editing with teletype/atom
## Stolen from sim.py
# Stuff all of the fundamental constants and commonly used parameters into a
# class. Any instance of this class will thus have all the constants.
class Params(object):
  """Stolen from sim.py. Expanded in this file.
  """
  def __init__(self):
    self.post_hook_func = None
    self.F = 96485  # Faraday constant [C/mol]
    self.R = 8.314  # Gas constant [J/K*mol]
    self.eo = 8.854e-12  # permittivity of free space [F/m]
    self.kb = 1.3806e-23  # Boltzmann constant [m2 kg/ s2 K1]
    self.q = 1.602e-19  # electron charge [C]
    self.tm = 7.5e-9  # thickness of cell membrane [nm]
    self.cm = 0.05  # patch capacitance of membrane [F/m2]
    self.T = 310  # temperature in Kelvin
    self.deltaGATP = -37000 # free energy released in ATP hydrolysis [J/mol]
    self.cATP = 1.5  # ATP concentration (mol/m3)
    self.cADP = 0.15  # ADP concentration (mol/m3)
    self.cPi = 0.15  # Pi concentration (mol/m3)
    self.alpha_NaK =1.0e-7 # max rate constant Na-K ATPase/unit surface area
    self.KmNK_Na = 12.0  # NaKATPase enzyme ext Na half-max sat value
    self.KmNK_K = 0.2  # NaKATPase enzyme ext K half-max sat value
    self.KmNK_ATP = 0.5  # NaKATPase enzyme ATP half-max sat value

    self.cell_r = 5.0e-6  # radius of single cell
    self.gj_len = 100e-9  # distance between two GJ connected cells [m]
    self.cell_sa = (4 * math.pi * self.cell_r ** 2)  # cell surface area
    self.cell_vol = ((4 / 3) * math.pi * self.cell_r ** 3)  # cell volume

    # Simulation control.
    self.sim_dump_interval=10
    self.sim_long_dump_interval=100
    self.no_dumps = False

    # Numerical-integration parameters. These place a limit on how much
    # any cell's Vmem, or any ion concentration, can change in one timestep.
    self.sim_integ_max_delt_Vm = .0001      # Volts/step
    # .001 means that no [ion] can change by more than .1% in one timestep.
    self.sim_integ_max_delt_cc = .001
    self.adaptive_timestep = True   # So that the params above get used.
    self.use_implicit = False

    ###JJJ My new neutral ion "A"
    self.concA_fixed_amount = 100   # moles/m3

class Planaria(Params) :
  """
  The Planaria class is a subclass of the Params class in sim.py. The
  Planaria class allows for some cell parameters to change and has methods
  for creating random mutations and crossovers.
  """
  ## A class variable to identify which parameters are modifiable by the algo.
  params = ['cell_r', 'gj_len', 'kM', 'N', 'scale', 'n_cells', 'cell_grow']
  cell_r_range = (2e-6, 1e-5)
  gj_len_range = (50e-9, 100e-8)
  num_cells_range = (2, 20)
  kM_range = (0.2, 2)
  N_range = (1, 20)
  scale_range = (0.1, 0.5)

  def __init__(self) :

    ## Initialize the constants and other parameters that are not changed by
    ## calling the super class (Params) from sim.py
    Params.__init__(self)
    self.n_cells = np.random.randint(*self.num_cells_range)
    self.n_GJs = self.n_cells - 1
    
    self.cell_r = get_random(self.cell_r_range)
    self.gj_len = get_random(self.gj_len_range)
    ## Recalculate the surface area and volume
    self.cell_sa = (4 * math.pi * self.cell_r ** 2)  # cell surface area
    self.cell_vol = ((4 / 3) * math.pi * self.cell_r ** 3)  # cell volume

    self.kM = get_random(self.kM_range)
    self.N = get_random(self.N_range)
    self.scale = get_random(self.scale_range)
    
    self._init_big_arrays(['M'])
    
    self.Na = self.ion_i['Na']
    self.K = self.ion_i['K']
    self.Cl = self.ion_i['Cl']
    self.P = self.ion_i['P']
    self.M = self.ion_i['M']

    fb_cells = [0, self.n_cells - 1]
    self.Dm_array[self.K, fb_cells] = 1.7e-17
    self.Dm_array[:, 1:-1] = 0
    
    for i in fb_cells :
      self.ion_magic[self.K, i] = self.magic_Hill_inv(self.M, self.N, self.kM, i)

    self.gj_connects['from'] = range(self.n_GJs)
    self.gj_connects['to'] = self.gj_connects['from'] + 1
    self.gj_connects['scale'] = self.scale

    self.z_array[self.M] = -1
    self.GJ_diffusion[self.M] = 1e-18

    self.cc_env[self.Na] = 145
    self.cc_cells[self.K] = 5
    self.cc_env[self.P] = 10
    self.cc_env[self.Cl] = 140
    
    self.cc_cells[self.Na] = 12
    self.cc_cells[self.K] = 139
    self.cc_cells[self.P] = 135
    self.cc_cells[self.Cl] = 15

    spread = 1
    spread = np.linspace(spread, -spread, self.n_cells)
    self.cc_cells[self.M, :] = 1 + spread
    self.cc_cells[self.Na, :] += spread

    self.gj_len = 15e-9

    self.GJ_diffusion[self.Na] = 1.33e-17
    self.GJ_diffusion[self.K] = 1.96e-17
    self.GJ_diffusion[self.M] = 1e-14
    self.GJ_diffusion[self.P] = 0
    
    self.Vm = self.compute_Vm()

    
  def _init_big_arrays(self, extra_ions=[]):
    """Stolen from sim.py, adapted to use as class method."""
    # ion properties (Name, base membrane diffusion [m2/s], valence
    #   initial concentration inside cell [mol/m3],
    #   fixed concentration outside cell [mol/m3],
    # These are temporary structures. We use them to provide initial values for
    # the big arrays we are about to build, and to specify the order of which
    # row represents which ion in those arrays.
    Na={'Name':'Na', 'D_mem':1e-18, 'D_GJ':1e-18, 'z':1, 'c_in':10, 'c_out':145}
    K ={'Name':'K',  'D_mem':1e-18, 'D_GJ':1e-18, 'z':1, 'c_in':125,'c_out':5}
    Cl={'Name':'Cl', 'D_mem':1e-18, 'D_GJ':1e-18, 'z':-1,'c_in':55, 'c_out':140}
    P= {'Name':'P',  'D_mem':0,     'D_GJ':1e-18, 'z':-1,'c_in':80, 'c_out':10}

    # stack the above individual dictionaries into a list to make it easier to
    # process them in the loop below.
    ions_vect = [Na, K, Cl, P]

    # Any particular sim may want to declare extra ions.
    for ion in extra_ions:
      ions_vect.append({'Name':ion, 'D_mem':0.0, 'D_GJ':1e-18,
                         'z':0, 'c_in':0,  'c_out':0})
    n_ions = len(ions_vect)

    self.cc_cells = np.empty((n_ions, self.n_cells))
    self.Dm_array = np.empty((n_ions, self.n_cells))
    self.z_array  = np.empty((n_ions))
    self.cc_env   = np.empty((n_ions))
    self.GJ_diffusion = np.empty((n_ions))

    self.ion_i    = {}

    # Push the parameters of the above ions into the various arrays.
    for row, ion_obj in enumerate(ions_vect):
      self.cc_cells[row,:] = ion_obj['c_in']       # initial cell conc
      self.cc_env  [row]   = ion_obj['c_out']      # fixed environmental conc
      self.Dm_array [row] = ion_obj['D_mem']       # initial membrane diff coeff
      self.z_array[row] = ion_obj['z']             # fixed ion valence
      self.GJ_diffusion[row] = ion_obj['D_GJ']     # diffusion rate through GJs
      self.ion_i[ion_obj['Name']] = row            # map ion name -> its row

    # Initialize the magic arrays to their default no-magic state.
    magic_dtype=np.dtype ([('type','i4'),('kM','f4'), ('N','f4'), \
                           ('cell','i4'), ('ion','i4'), ('cell2','i4')])
    self.ion_magic = np.zeros ((n_ions, self.n_cells), dtype=magic_dtype)
    self.GJ_magic  = np.zeros ((self.n_GJs), dtype=magic_dtype)
    self.gen_magic = np.zeros ((n_ions, self.n_cells), dtype=magic_dtype)

    # Create default arrays for GJs, and for generation, decay rates.
    self.gj_connects=np.zeros((self.n_GJs), dtype=[('from','i4'),('to','i4'),('scale','f4')])
    self.gen_cells   = np.zeros ((n_ions, self.n_cells))
    self.decay_cells = np.zeros ((n_ions))

  def resize_params(self) :
    
    (f, i) = self.cc_cells.shape
    self.n_GJs = self.n_cells - 1
    # Re-Initialize the magic arrays to their default no-magic state with a new cell.
    magic_dtype=np.dtype ([('type','i4'),('kM','f4'), ('N','f4'), \
                           ('cell','i4'), ('ion','i4'), ('cell2','i4')])
    x, _ = self.ion_magic.shape
    ## Only need to change the number of cells/gap junctions.
    ion_magic = np.zeros ((x, self.n_cells), dtype=magic_dtype)
    #print(self.GJ_magic.shape)
    GJ_magic  = np.zeros ((self.n_GJs), dtype=magic_dtype)
    #print(GJ_magic.shape)
    gen_magic = np.zeros ((x, self.n_cells), dtype=magic_dtype)
    diff = abs(i - self.n_cells)
    new_cc = np.zeros((f, i + diff))
    new_Dm = np.zeros((f, i + diff))
    for f in range(f) :
      for i in range(i) :
        new_cc[f][i] = self.cc_cells[f][i]
        new_Dm[f][i] = self.Dm_array[f][i]
        ion_magic[f][i] = self.ion_magic[f][i]
        gen_magic[f][i] = self.gen_magic[f][i]
    print(self.GJ_magic.shape)
    g, = self.GJ_magic.shape
    
    for g in range(g) :
      GJ_magic[g] = self.GJ_magic[g]

    self.cc_cells = new_cc
    self.Dm_array = new_Dm
    self.ion_magic = ion_magic
    self.GJ_magic = GJ_magic
    self.gen_magic = gen_magic

    # Create default arrays for GJs, and for generation, decay rates.
    self.gj_connects=np.zeros((self.n_GJs), dtype=[('from','i4'),('to','i4'),('scale','f4')])
    x, _ = self.gen_cells.shape
    self.gen_cells   = np.zeros((x, self.n_cells))
    self.Vm = self.compute_Vm()
    
  def register_post_hook(self, func) :
    self.post_hook_func = func

  # The main "do-it" simulation function.
  # Takes the current cc_cells[n_ions,n_cells], does all of the physics work, and
  # returns an array of concentration slew rates [n_ions,n_cells]; i.e.,
  # moles/m3 per second.
  # In normal operation, sim_slopes() is only called from sim.sim() and thus is
  # not needed outside of sim.py. However, the solve() functions in main.py call
  # sim_slopes() directly -- sim_slopes() == vector of zeroes indicates steady
  # state.
  def sim_slopes(self, t):

    num_cells = self.cc_cells.shape[1]
    self.Vm = self.compute_Vm()

    # General note: our units of flux are moles/(m2*s). The question: m2 of
    # what area? You might think that for, e.g., ion channels, it should be per
    # m2 of ion-channel area -- but it's not. All fluxes are per m2 of cell-
    # membrane area. Thus, the (e.g.,) diffusion rate through ion channels must
    # be scaled down by the fraction of membrane area occupied by channels.
    # The same goes for ion pumps and GJs.
    slew_cc = np.zeros (self.cc_cells.shape) # Per-ion cell fluxes
    # Run the Na/K-ATPase ion pump in each cell.
    # Returns two 1D arrays[N_CELLS] of fluxes; units are moles/(m2*s)
    f_Na, f_K, _ = stb.pumpNaKATP(self.cc_cells[self.ion_i['Na']],
                                  self.cc_env[self.ion_i['Na']],
                                  self.cc_cells[self.ion_i['K']],
                                  self.cc_env[self.ion_i['K']],
                                  self.Vm,
                                  self.T,
                                  self,
                                  1.0)

    # Kill the pumps on worm-interior cells (based on Dm=0 for all ions)
    keep_pumps = np.any(self.Dm_array>0, 0) # array[n_cells]
    f_Na *= keep_pumps
    f_K  *= keep_pumps

    # Update the cell-interior [Na] and [K] after pumping (assume env is too big
    # to change its concentration).
    slew_cc[self.ion_i['Na']] = f_Na
    slew_cc[self.ion_i['K']]  = f_K

    # Get the gap-junction Thevenin-equivalent circuits for all ions at once.
    # We get two arrays of [n_ions,n_GJs].
    # Units of Ith are mol/(m2*s); units of Gth are mol/(m2*s) per Volt.
    (GJ_Ith, GJ_Gth) = self.GJ_norton()

    # for each ion: (sorted to be in order 0,1,2,... rather than random)
    for ion_name,ion_index in sorted(self.ion_i.items(),key=operator.itemgetter(1)):
      # GHK flux across membranes into the cell
      # It returns array[N_CELLS] of moles/(m2*s)
        f_ED = self.GHK(ion_index)
        f_ED *= self.eval_magic(self.ion_magic[ion_index,:])
        slew_cc[ion_index] += f_ED

    # Gap-junction computations. Note the units of the Thevenin-equivalent
    # circuits; the "Ithev" is actually moles/(m2*s), just like f_gj.
    # These arrays are all [n_GJ].
    deltaV_GJ = (self.Vm[self.gj_connects['to']] - self.Vm[self.gj_connects['from']])
    f_gj = GJ_Ith + deltaV_GJ*GJ_Gth

    magic = self.eval_magic(self.GJ_magic)
    f_gj *= magic

    # Update cells with gj flux:
    # Note that the simple slew_cc[ion_index, gj_connects['to']] += f_gj
    # doesn't actually work in the case of two GJs driving the same 'to'
    # cell. Instead, we use np.add.at().
    for ion_name,ion_index in sorted(self.ion_i.items(),key=operator.itemgetter(1)):
        np.add.at(slew_cc[ion_index,:], self.gj_connects['from'], -f_gj[ion_index])
        np.add.at(slew_cc[ion_index,:], self.gj_connects['to'],    f_gj[ion_index])

    # The current slew_cc units are moles/(m2*s), where the m2 is m2 of
    # cell-membrane area. To convert to moles/s entering the cell, we multiply
    # by the cell's surface area. Then, to convert to moles/m3 per s entering
    # the cell, we divide by the cell volume.
    slew_cc *= (self.cell_sa / self.cell_vol)

    # Next, do generation and decay.
    for ion_name,ion_index in sorted(self.ion_i.items(),key=operator.itemgetter(1)):
      gen = self.gen_cells[ion_index,:] * self.eval_magic(self.gen_magic[ion_index,:])
      decay = self.cc_cells[ion_index,:] * self.decay_cells[ion_index]
      slew_cc[ion_index] += gen - decay

    if (self.post_hook_func != None):
      self.post_hook_func(t, self, cc_cells, slew_cc)

    return slew_cc    # Moles/m3 per second.

  # Given: per-cell, per-ion charges in moles/m3.
  # First: sum them per-cell, scaled by valence to get "signed-moles/m3"
  # Next: multiply by F to convert moles->coulombs. Multiply by cell volume/
  # surface area to get coulombs/m2, and finally divide by Farads/m2.
  # The final scaling factor is F * p.cell_vol / (p.cell_sa*p.cm),
  # or about 3200 mV per (mol/m3)
  def compute_Vm(self):
    # Calculate Vmem from scratch via the charge in the cells.
    rho_cells = (self.cc_cells * self.z_array[:,np.newaxis]).sum(axis=0) * self.F
    return (rho_cells * self.cell_vol / (self.cell_sa*self.cm))

  def GHK(self, ion_index):
    num_cells = self.cc_cells.shape[1]
    f_ED = stb.electroflux(self.cc_env[ion_index] * np.ones(num_cells),
                           self.cc_cells[ion_index],
                           self.Dm_array[ion_index],
                           self.tm * np.ones(num_cells),
                           self.z_array[ion_index] * np.ones(num_cells),
                           self.Vm,
                           self.T,
                           self,
                           rho=np.ones(num_cells))
    return (f_ED)

  def sim_implicit(self, end_time):
    import scipy
    #global cc_cells, Vm
    num_ions, num_cells = self.cc_cells.shape
    
    def wrap (t, y):
      #global cc_cells
      #print ('----------------\nt={:.9g}'.format(t))
      cc = self.cc_cells
      self.cc_cells = y.reshape(num_ions, num_cells)
      np.nan_to_num(self.cc_cells, copy=False)
      slew_cc = self.sim_slopes(t) # moles/(m3*s)
      #print(slew_cc)
      slew_cc = slew_cc.reshape(num_ions*num_cells)
      #np.set_printoptions (formatter={'float':'{:6.2f}'.format},linewidth=120)
      #print ('y={}'.format(y))
      #np.set_printoptions (formatter={'float':'{:7.2g}'.format},linewidth=120)
      #print ('slews={}'.format(slew_cc))
      #print(slew_cc)
      
      np.nan_to_num(slew_cc, copy=False)
      self.cc_cells = cc
      return slew_cc

    # Save information for plotting at sample points. Early on (when things
    # are changing quickly) save lots of info. Afterwards, save seldom so
    # as to save memory. So, 100 points in t=[0,50], then 200 in [50, end_time].
    boundary=min(50,end_time)
    t_eval = np.linspace(0,boundary,50,endpoint=False)
    if (end_time>50):
      t_eval = np.append(t_eval, np.linspace (boundary, end_time, 200))

    # run the simulation loop:
    y0 = self.cc_cells.reshape(num_ions*num_cells)
    #print(y0)
    #print(t_eval)
    
    bunch = scipy.integrate.solve_ivp(wrap, (0, end_time), y0, method='BDF', \
                                      t_eval=t_eval)

    #print ('{} func evals, status={} ({}), success={}'.format \
    #       (bunch.nfev, bunch.status, bunch.message, bunch.success))
    t_shots = t_eval.tolist()
    # bunch.y is [n_ions*n_cells, n_timepoints]
    cc_shots = [y.reshape((num_ions,num_cells)) for y in bunch.y.T]
    self.cc_cells = cc_shots[-1]
    return (t_shots, cc_shots)

  # Builds and returns a Norton equivalent model for all GJs.
  # Specifically, two arrays GJ_Ith and GJ_Gth of [n_ions,n_GJ].
  # Ith[i,g] is the diffusive flux of ion #i in the direction of GJ[g].from->to,
  # and has units (mol/m2*s)
  # Gth*(Vto-Vfrom) is the drift flux of particles in the from->to direction;
  # Gth has units (mol/m2*s) per Volt.
  def GJ_norton (self):
    #global cc_cells
    n_GJ = self.gj_connects.size
    n_ions = self.cc_env.size
    
    GJ_Ith = np.empty((n_ions, n_GJ))
    GJ_Gth = np.empty((n_ions, n_GJ))

    # Compute ion drift and diffusion through GJs. Assume fixed GJ spacing
    # of gj_len between connected cells.
    # First, compute d_conc/dx (assume constant conc in cells, and constant
    # gradients in the GJs).
    GJ_from = self.gj_connects['from']       # Arrays of [n_GJ,1]
    GJ_to   = self.gj_connects['to']
    D_scale = self.gj_connects['scale']
    
    for ion_index in range(n_ions):
      deltaC_GJ = (self.cc_cells[ion_index,GJ_to] - self.cc_cells[ion_index,GJ_from]) \
        / self.gj_len
      
      # Assume that ion concentration for any ion is constant within a cell,
      # and then transitions linearly across a GJ. Then c_ave[g] is the conc
      # of the current ion, in the middle of GJ #g. Why do we care? Because
      # when we compute flux = velocity * concentration, then this is the
      # concentration that we will use (regardless of which direction the
      # drift current is actually flowing).
      c_avg = (self.cc_cells[ion_index,GJ_to] + self.cc_cells[ion_index,GJ_from]) / 2

      # Finally, electrodiffusive gj flux:
      # f_gj[i] is flux (moles/(m2*s)), in the direction from GJ input to
      # output. Note that D/kT gives the drift mobility.
      D = self.GJ_diffusion[ion_index] * D_scale
      alpha = (c_avg * self.q * self.z_array[ion_index]) * (D/(self.kb*self.T*self.gj_len))
      GJ_Ith[ion_index,:] = -D*deltaC_GJ
      GJ_Gth[ion_index,:] = -alpha

    return (GJ_Ith, GJ_Gth)

  # Takes an array of magic_dtype. It is either [N_CELLS] (for ion-channel
  # magic) or [N_GJs] (for GJ magic).
  # Returns a same-sized array of scalar scale factors in [0,1].
  def eval_magic (self, magic_arr):
    #global Vm

    type = magic_arr['type']    # Magic_arr is a structured array; break it
    kM   = magic_arr['kM']      # into a separate simple array for each field.
    N    = magic_arr['N']
    cell = magic_arr['cell']
    ion  = magic_arr['ion']
    cell2= magic_arr['cell2']
    
    # The default is type==0, which results in scale=1 (i.e., no scaling)
    use_Vmem = np.flatnonzero (type==1) # indices of cells using Vmem
    buf_ion  = np.flatnonzero (type==2) # Hill buffer with ions
    Hill_ion = np.flatnonzero (type>=2) # Hill inv or buf with ions
    
    # Some advanced-indexing trickery for the use-ion-conc case (i.e., the
    # channels whose input is a concentration and not Vm).
    # Say that cells #3 and #5 are using ion concentrations. Then, using the
    # ion[] and cell[] arrays just above,
    #   cell #3 takes its input from ion number ion[3] (in cell number cell[3])
    #   cell #5 takes its input from ion number ion[5] (in cell number cell[5])
    # and we must set inps[3] = cc_cells[ion[3],cell[3]]
    #                 inps[5] = cc_cells[ion[5],cell[5]]
    inps  = np.empty (magic_arr.size)   # Temporary array to build inputs
    inps[Hill_ion] = self.cc_cells[ion[Hill_ion], cell[Hill_ion]]

    # if (ligand) -> MM (kM, N, cell, ion) and 1- if needed
    # Implement the Hill buffer or inverter function (still for use-ion-conc).
    scale = np.ones  (magic_arr.size)   # Final array to return to the caller
    scale[Hill_ion] = 1 / (1 + ((inps[Hill_ion]/kM[Hill_ion])**N[Hill_ion]))
    scale[buf_ion] = 1 - scale[buf_ion]
    
    # And similar, but even trickier, for cells with channels that use Vmem
    # The trick is the cell = -1 means that the cell gets Vm=0
    # if (V) -> 1 / 1+exp(N*(v1-v2-kM))
    V1 = np.empty(magic_arr.size)
    V2 = np.empty(magic_arr.size)
    V1[use_Vmem] = self.Vm[cell [use_Vmem]]
    V2[use_Vmem] = self.Vm[cell2[use_Vmem]]
    V1_is0V_idx = np.flatnonzero (cell ==-1) # the i such that cell[i]== -1
    V2_is0V_idx = np.flatnonzero (cell2==-1)
    V1[V1_is0V_idx] = 0
    V2[V2_is0V_idx] = 0
    inps [use_Vmem] = V1[use_Vmem] - V2[use_Vmem] - kM[use_Vmem]
    scale[use_Vmem] = 1 / (1 + np.exp(N[use_Vmem]*inps[use_Vmem]))
    
    return (scale)

  def magic_Hill_buf (self, input_ion, N, kM, input_cell):
    return ((2, kM, N, input_cell, input_ion, 0))

  def magic_Hill_inv (self, input_ion, N, kM, input_cell):
    return ((3, kM, N, input_cell, input_ion, 0))


  def do_mutation(self) :
    """Performs a point-wise mutation of a parameter in the planaria.
    The parameter that is mutated is randomly chosen. The value of the parameter
    is scaled by a random factor between 0.5 and 2.
    """
    random_params = self._get_random_params()
    for param in random_params :
      print(param)
      factor = get_random((0.5, 2))
      if param =='cell_grow':
        print("Growing")
        x = getattr(self, 'n_cells')
        setattr(self, 'n_cells', x + 1)
        self.resize_params()
        return
      x = getattr(self, param)
      if param == 'n_cells' :
        setattr(self, param, int(x * factor))
        return
      setattr(self, param, x * factor)
      if param == 'cell_r' :
        ## Need to recalculate these when the radius changes.
        self.cell_sa = (4 * math.pi * self.cell_r ** 2)  # cell surface area
        self.cell_vol = ((4 / 3) * math.pi * self.cell_r ** 3)  # cell volume


  def do_crossover(self, other_planaria) :
    """Performs a crossover mutation between this planaria and another planaria.
    A random set of parameters are used to generate 2 new Planaria classes one
    for each of the resulting children.
    """

    random_params = self._get_random_params()
    child1, child2 = deepcopy(self), deepcopy(other_planaria)
    for param in random_params :
      if param == 'cell_grow' or param == 'n_cells':
        continue
      setattr(child1, param, getattr(other_planaria, param))
      setattr(child2, param, getattr(self, param))
      if param =='cell_r' :
        ## Need to recalculate these when the radius changes.
        child1.cell_sa = (4 * math.pi * child1.cell_r ** 2)  # cell surface area
        child1.cell_vol = ((4 / 3) * math.pi * child1.cell_r ** 3)  # cell volume
        child2.cell_sa = (4 * math.pi * child2.cell_r ** 2)  # cell surface area
        child2.cell_vol = ((4 / 3) * math.pi * child2.cell_r ** 3)  # cell volume
        
    return child1, child2


  def run(self, time_steps) :
    """Runs the model for time_steps number of iterations. Calculates and returns
    the fitness selection metric.
    """
    print(time_steps)
    self.t_shots, self.cc_shots = self.sim_implicit(time_steps)
    self.vm = self.compute_Vm()
    return self.vm

  def fitness(self) :
    """Calculates the fitness selection metrics from the last run of the planaria.
    """
    ## Get the Vmem difference between each adjacent cell in the planaria
    ## Return the max_diff of the planaria
    min_vmem = min(self.vm)
    max_vmem = max(self.vm)
    
    return abs(min_vmem - max_vmem)


  def _get_random_params(self) :
    """Chooses random parameters from the Planaria class. Returned as a list.
    """
    num = random.randint(1, len(self.params))
    choices = []
    for _ in range(num) :
      choice = random.choice(self.params)
      choices.append(choice)
    return choices


def get_random(parameter_range) :
  """Takes in a tuple of (minimum, maximum) or a single value. If a single value
  is given, the return is a random number between 1 and the value. Otherwise,
  the return is a random number between the minimum and maximum of the supplied
  range.
  """
  return random.uniform(*parameter_range)


class Evolve(object) :
  """Simulates the evolution of planaria from a random start point using a genetic
  algorithm. The results are plotted in a matrix with 1 planaria per row.
  """
  dead_threshold = 0.1
  mutation_threshold = 0.3

  def __init__(self, number_of_planaria) :
    """number_of_planaria is the number of planaria to initialize.
    """
    self.num_of_planaria = number_of_planaria
    self.max_cells = 3
    ## Call some initialization functions to set everything up.
    planaria = []
    for _ in range(number_of_planaria) :
      p = Planaria()
      planaria.append(p)
      if p.n_cells > self.max_cells :
        self.max_cells = p.n_cells
        
    self.planaria_list = planaria
    ## Open and initialize the graph.
    self._init_graph()

  def start(self, epochs, steps_per_mutation) :
    """Starts the evolution.
    epochs is the total number of rounds of the algorithm to take.
    steps_per_mutation is the number of steps to take before applying selection
    criteria.
    """
    while(epochs > 0) :
      ## Perform one step.
      epochs -= 1
      ## Update the graph after each planaria is run for steps_per_mutation time.
      for index, planaria in enumerate(self.planaria_list) :
        if planaria.n_cells > self.max_cells :
          print(planaria.n_cells)
          self.max_cells = planaria.n_cells
          self._resize_graph(self.max_cells)
        
        data = planaria.run(steps_per_mutation)
        print(data)
        self._update_graph(index, data)
        self._show_graph()

      good, bad = self._cull_planaria()
      self._reproduce(good, bad)
      self._apply_mutation()
      

  def _reproduce(self, good_list, bad_list) :
    while len(bad_list) > 1:
      choices = len(good_list)
      [i1, i2] = random.choice(choices, 2)
      plan1 = self.planaria_list[good_list[i1][0]]
      plan2 = self.planaria_list[good_list[i2][0]]

      c1, c2 = plan1.do_crossover(plan2)

      [b1, b2] = bad_list[:2]
      bad_list= bad_list[2:]
      self.planaria_list[b1[0]] = c1
      self.planaria_list[b2[0]] = c2
      
      
  def _apply_mutation(self) :
    """Choose a random number of planaria to undergo mutations.
    """
    n = random.randint(self.num_of_planaria)
    choices = random.choice(self.planaria_list, n)
    for mut in choices :
      mut.do_mutation()
      

  def _cull_planaria(self) :
    """Gets and ranks the planaria by the selection criteria.
    The worst performing planaria are culled and replaced with cross-over mutations.
    """
    planaria_rank_list = []
    for i, planaria in enumerate(self.planaria_list):
      planaria_fit = planaria.fitness()
      planaria_rank_list.append((i, planaria_fit))
    
    planaria_rank_list.sort(key=lambda x: x[1])
    n = (len(planaria_rank_list))//2
    good_list = planaria_rank_list[n:]
    bad_list = planaria_rank_list[:n]
    return (good_list, bad_list)
    

  def _init_graph(self) :
    """Sets up and opens the window to graph the evolving lifecycle of the planaria.
    Creates an instance of the Board class.
    """
    self.graph = Board(self.num_of_planaria, self.max_cells)


  def _update_graph(self, row, data) :
    """Updates the graph with new information.
    """
    self.graph.update_row(row, data)

  def _show_graph(self) :
    self.graph.show_graph()

  def _resize_graph(self, row) :
    self.graph.resize_graph(int(row))
    

class Board(object) :
  def __init__(self, max_rows, max_cols) :
    """Initializes the board to the given number of rows and columns.
    Rows corresponds to the maximum number of cells in any Planaria.
    Columns correspons to the number of Planaria to plot.
    """
    self.rows = max_rows
    self.cols = max_cols
    self.board = np.zeros((max_rows, max_cols))
    plt.ion()

    colors = mcolors.Normalize(vmin=-.1, vmax=0)
    
    self.opts = {'rasterized':True, 'cmap':'viridis', 'norm':colors}
    self.figure, self.ax = plt.subplots()
    self.mat = self.ax.pcolormesh(self.board, **self.opts)
    self.figure.colorbar(self.mat, ax=self.ax)
    self.ax.set_autoscalex_on(True)
    plt.pause(0.001)
    
  def update_row(self, row, data) :
    """Updates a single row in the graph to the supplied data.
    Data must be an array of values.
    """
    if len(data) < self.cols :
      new_arr = np.zeros(self.cols)
      for i, d in enumerate(data) :
        new_arr[i] = d
      data = new_arr
    data = np.asarray(data)
    for i in range(self.cols) :
      self.board[row][i] = data[i]

  def show_graph(self) :
    self.mat = self.ax.pcolormesh(self.board, **self.opts)
    plt.pause(0.001)

  def resize_graph(self, cols) :
    """Resizes the graph to the new number of rows and cols.
    Zeros are used in all new entries. The figure can only grow in size.
    """
    ## These rows and columns may be messed up.
    #if rows >= self.rows and cols >= self.cols :
    new_board = np.zeros((self.rows, cols))
    print(new_board.shape)
    print(self.board.shape)
    for r in range(self.rows) :
      for c in range(self.cols) :
        new_board[r][c] = self.board[r][c]
    self.board = new_board
    self.cols = cols
    self.ax.pcolormesh(self.board, **self.opts)
    plt.pause(0.001)
      
      
def main(epochs, steps_per_epoch, number_of_planaria) :
  program = Evolve(number_of_planaria)
  program.start(epochs, steps_per_epoch)

  
if __name__ == "__main__" :
  import sys
  args = sys.argv
  if len(args) == 4:
    epochs = int(args[1])
    steps_per_epoch = int(args[2])
    number_of_planaria = int(args[3])
  else :
    epochs = 10
    steps_per_epoch = 10
    number_of_planaria = 10
  main(epochs, steps_per_epoch, number_of_planaria)
  
    
