# -*- coding: UTF-8 -*-
from agents.base import SequentialFineTune
from agents.er import ExperienceReplay
from agents.ewc import EWC
from agents.lwf import LwF
from agents.mas import MAS
from agents.si import SI
from agents.dt2w import DT2W
from agents.aser import ASER
from agents.herding import Herding
from agents.inversion import Inversion
from agents.clops import CLOPS
from agents.der import DarkExperienceReplay
from agents.gr import GenerativeReplay
from agents.er_sub import ER_on_Subject


agents = {'SFT': SequentialFineTune,
          'ER': ExperienceReplay,
          'ER_Sub': ER_on_Subject,
          'EWC': EWC,
          'LwF': LwF,
          'SI': SI,
          'MAS': MAS,
          'DT2W': DT2W,
          'ASER': ASER,
          'Herding': Herding,
          'Mnemonics': Herding,
          'Inversion': Inversion,
          'CLOPS': CLOPS,
          'DER': DarkExperienceReplay,
          'GR': GenerativeReplay,
          }

agents_replay = ['ER', 'DER', 'Herding', 'Mnemonics', 'ASER', 'Inversion', 'CLOPS', 'GR', 'ER_Sub']
