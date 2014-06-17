import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import os, os.path as op
import nipype.algorithms.misc as misc
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.pipeline.engine as pe          # pypeline engine
from nipype.interfaces.utility import Function

import smtplib

#itisfinishedprocessing@gmail.com.
def send_email(TO=["erik.sweed@gmail.com"],SUBJECT="DONE", TEXT="Finished"):
        SERVER = "smtp.gmail.com:587"
        username = "itisfinishedprocessing"
        password = "throwawaypassword"

        FROM = "itisfinishedprocessing@gmail.com"

        #TEXT = "This message was sent with Python's smtplib."

        # Prepare actual message

        message = """\
        From: %s
        To: %s
        Subject: %s

        %s
        """ % (FROM, ", ".join(TO), SUBJECT, TEXT)

        # Send the mail

        server = smtplib.SMTP(SERVER)
        server.starttls()
        server.login(username,password)
        server.sendmail(FROM, TO, message)
        server.quit()




def calc_tpm_fn(tracks):
   import os
   from nipype import logging
   from nipype.utils.filemanip import split_filename
   path, name, ext = split_filename(tracks)
   file_name = os.path.abspath(name + 'TPM.nii')
   iflogger = logging.getLogger('interface')
   iflogger.info(tracks)
   import subprocess
   iflogger.info(" ".join(["tracks2prob","-vox", "1.0", "-totallength", tracks, file_name]))
   subprocess.call(["tracks2prob", "-vox", "1.0", "-totallength", tracks, file_name])
   return file_name

calc_tpm = pe.Node(name='calc_tpm',
               interface=Function(input_names=["tracks"],
                                  output_names=['tpm'],
                                  function=calc_tpm_fn))

fsl.FSLCommand.set_default_output_type('NIFTI')

info = dict(tracks=[['subject_id', '*_CSD_tracked*']],
            tdi=[['subject_id','*_TDI*']])

control_list = ['p07090', 'p07108',
'p07113', 'p07116', 'p07131', 'p07183', 'p07188', 'p07198',
'p07200', 'p07232', 'p07242', 'p07248', 'p07262', 'p07305',
'p07465', 'p07467', 'p07468', 'p07488', 'p07493', 'p07509',
'p07519', 'p07523', 'p07535', 'p07601', 'p07612', 'p07663']

patient_list = [
'p06316', 'p06871', 'p06873', 'p06889', 'p06890',
'p06891', 'p06904', 'p06905', 'p06933', 'p06940',
'p06941', 'p06968', 'p07091', 'p07109', 'p07153',
'p07155', 'p07258', 'p07276', 'p07594', 'p07599',
'p07602', 'p07611', 'p07616', 'p07618', 'p07677',
'p07685', 'p07194']

control_list.extend(patient_list)
subject_list = control_list
#subject_list = ['p06316']

output_dir = '/media/EBS/TPM_APM'
data_dir = '/mnt/TDI_lmax8/parkflow_tdis/parkflow_tdis'

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name = 'datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_dir
datasource.inputs.field_template = dict(tracks='*%s/CSDstreamtrack/%s', tdi='*%s/tdi_native_to_nii/%s')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = False


divide_tdi_by_tpm = pe.Node(interface=fsl.MultiImageMaths(), name="divide_tdi_by_tpm")
divide_tdi_by_tpm.inputs.op_string = "-div %s"

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.overwrite = True

graph = pe.Workflow(name='tpm_apm')
graph.base_dir = output_dir
graph.connect([(infosource, datasource,[('subject_id', 'subject_id')])])
graph.connect([(datasource, calc_tpm,[('tracks', 'tracks')])])

graph.connect([(calc_tpm, divide_tdi_by_tpm,[('tpm', 'in_file')])])
graph.connect([(datasource, divide_tdi_by_tpm,[('tdi', 'operand_files')])])

graph.connect([(divide_tdi_by_tpm, datasink, [("out_file", "@subject_id.apm")])])
graph.connect([(calc_tpm, datasink, [("tpm", "@subject_id.tpm")])])
graph.connect([(infosource, datasink,[('subject_id','@subject_id')])])
#graph.run(updatehash=False)#plugin='MultiProc', plugin_args={'n_procs' : 3})
graph.run(plugin='MultiProc', plugin_args={'n_procs' : 32})
send_email()
