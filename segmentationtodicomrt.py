from lungmask import mask
import SimpleITK as sitk
import pydicom as dicom
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
import os

from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib.pyplot as plt

'''-------------------------------------------------------------------
    Mask
'''
def Image2Mask(path,file_name):
    input_image = sitk.ReadImage(path+'dicom/'+file_name)
    segmentation = mask.apply(input_image)

    x = segmentation[0]
    img = sitk.GetImageFromArray(x)
    sitk.WriteImage(img, path+'/Mask/Mask-'+file_name)
    
    
    
'''-------------------------------------------------------------------
    BorderPixels2NumpyArray
'''
def isBoarder(i,j,val,num):
    if num[i,j]==val and sum(sum(num[i-1:i+2,j-1:j+2]==val))<9:
        isBoarder=True
    else:
        isBoarder=False
    return isBoarder

def BorderPixels2NumpyArray(path,file_name,region_number):
    ds = dicom.read_file(path+'mask/'+'Mask-'+file_name, force=True)
    num=ds.pixel_array
    
    (bi,bj)=num.shape
    
    # Find Border pixels of each region
    #for region_number in range(2):
    val=region_number
    fi=-1
    fj=-1
    sw=False
    for i in range(bi):
        if sw:
            break
        for j in range(bj):
            if num[i,j]==val:
                fi=i
                fj=j
                #print(i,j,num[i,j])
                sw=True
                break
    i=fi
    j=fj
    #print(i,j,num[i,j])

    # Create numpy array of borders cordinations 
    meet=np.ones(num.shape)
    li=i
    lj=j
    meet[i,j]=0
    borders=[]
    a=1
    while a<2000:
        borders.append([i,j])
        i=li
        j=lj
        a=a+1
        #print(a)
        if num[i+1,j]==val and isBoarder(i+1,j,val,num) and meet[i+1,j]:
            li=i+1
            lj=j
            #print(li,lj,num[li,lj])
            meet[li,lj]=0

        elif num[i+1,j+1]==val and isBoarder(i+1,j+1,val,num) and meet[i+1,j+1]:
            li=i+1
            lj=j+1
            #print(li,lj,num[li,lj])
            meet[li,lj]=0
        elif num[i,j+1]==val and isBoarder(i,j+1,val,num)and meet[i,j+1]:
            li=i
            lj=j+1
            #print(li,lj,num[li,lj])
            meet[li,lj]=0
        elif num[i-1,j+1]==val and isBoarder(i-1,j+1,val,num)and meet[i-1,j+1]:
            li=i-1
            lj=j+1
            #print(li,lj,num[li,lj])
            meet[li,lj]=0

        elif num[i-1,j]==val and isBoarder(i-1,j,val,num)and meet[i-1,j]:
            li=i-1
            lj=j
            #print(li,lj,num[li,lj])
            meet[li,lj]=0

        elif num[i+1,j-1]==val and isBoarder(i+1,j-1,val,num)and meet[i+1,j-1]:
            li=i+1
            lj=j-1
            #print(li,lj,num[li,lj])
            meet[li,lj]=0
        elif num[i,j-1]==val and isBoarder(i,j-1,val,num)and meet[i,j-1]:
            li=i
            lj=j-1
            #print(li,lj,num[li,lj])
            meet[li,lj]=0
        elif num[i-1,j-1]==val and isBoarder(i-1,j-1,val,num)and meet[i-1,j-1]:
            li=i-1
            lj=j-1
            #print(li,lj,num[li,lj])
            meet[li,lj]=0
        if (li==i and lj==j):
            break
    borders.append([fi,fj])

    # Shapenning borders pixels 
    for t in range(len(borders)):
        #print(t,borders[t])
        i=borders[t][0]
        j=borders[t][1]
        num[i,j]=100

    file_name=file_name.replace('mask-','') 
    np.save(path+ 'borders/Border'+str(val)+'-'+file_name.replace('Mask-','') +'.npy', borders)

'''
    Text to BorderPixels
'''
def BorderAlphabet2Numpy(path,file_name,region_number):
    #ds = dicom.read_file(path+'mask/'+file_name, force=True)
    print(path+'TextImage/'+file_name)
    img=Image.open(path+'TextImage/'+file_name)
    img2 = img.convert("P")

    #img2 = Image.open(fname).convert('L')
    #img2 = np.asarray(img2)
    labeled, nr_objects = ndimage.label(img2) 

    plt.imshow(img2)

    Nimg=np.logical_not(img2)
    #Nimg=Nimg[:,:,0]

    plt.imshow(Nimg)

    l1, nr_objects = ndimage.label(Nimg) 
    print("Number of objects is {}".format(nr_objects))
    # Number of objects is 4 
    nl=(l1>1)
    l2=nl*np.ones(l1.shape)
    l3=l2*l1
    l4=l3+nl*np.ones(l1.shape)*(labeled.max()-1)
    plt.imshow(l4)

    l_all=labeled+l4
    plt.imshow(l_all)    

    num = np.array(l_all)   

    print('number of objects',int(l_all.max()))
    
    (bi,bj)=num.shape
    #print(bi,bj)
    # Find Border pixels of each region
    #for region_number in range(2):
    val=region_number
    fi=-1
    fj=-1
    sw=False
    for i in range(bi):
        if sw:
            #print('break')
            break
        for j in range(bj):
            #print(i,j,num[i,j])
            if num[i,j]==val:
                fi=i
                fj=j
                #print(i,j,num[i,j])
                sw=True
                break
    i=fi
    j=fj
    print('initial points',i,j,num[i,j])
    print('val:',val)
    print(np.sum(num==val))
    # Create numpy array of borders cordinations 
    meet=np.zeros(num.shape)
    li=i
    lj=j
    #meet[i,j]=0
    borders=[]
    a=1
    while a<2000:
        borders.append([i,j])
        i=li
        j=lj
        a=a+1
        print(a)
        if num[i+1,j]==val and isBoarder(i+1,j,val,num) and not meet[i+1,j]:
            li=i+1
            lj=j
            print(li,lj,num[li,lj])
            meet[li,lj]=a

        elif num[i+1,j+1]==val and isBoarder(i+1,j+1,val,num) and not meet[i+1,j+1]:
            li=i+1
            lj=j+1
            print(li,lj,num[li,lj])
            meet[li,lj]=a
        elif num[i,j+1]==val and isBoarder(i,j+1,val,num)and not meet[i,j+1]:
            li=i
            lj=j+1
            print(li,lj,num[li,lj])
            meet[li,lj]=a
        elif num[i-1,j+1]==val and isBoarder(i-1,j+1,val,num)and not meet[i-1,j+1]:
            li=i-1
            lj=j+1
            print(li,lj,num[li,lj])
            meet[li,lj]=a

        elif num[i-1,j]==val and isBoarder(i-1,j,val,num)and not meet[i-1,j]:
            li=i-1
            lj=j
            print(li,lj,num[li,lj])
            meet[li,lj]=a

        elif num[i+1,j-1]==val and isBoarder(i+1,j-1,val,num)and not meet[i+1,j-1]:
            li=i+1
            lj=j-1
            print(li,lj,num[li,lj])
            meet[li,lj]=a
        elif num[i,j-1]==val and isBoarder(i,j-1,val,num)and not meet[i,j-1]:
            li=i
            lj=j-1
            print(li,lj,num[li,lj])
            meet[li,lj]=a
        elif num[i-1,j-1]==val and isBoarder(i-1,j-1,val,num)and not meet[i-1,j-1]:
            li=i-1
            lj=j-1
            print(li,lj,num[li,lj])
            meet[li,lj]=a
        if (li==i and lj==j):
            [i,j]=borders.pop(-1)
            num[i,j]=val+1
            if len(borders):
                [i,j]=borders.pop(-1)
            li=i
            lj=j
            print('------------------------a:',len(borders))
            print('Del-----',i,j)
        m=meet[li-1:li+2,lj-1:lj+2]
        print(a,m)
        if a>5 and np.sum((m<4) & (m>0)):
            li=fi
            lj=fj
            print (meet[li-1:li+1,lj-1:lj+1])
            print('++++++++++++++++++++++setting first point')
        if (fi==li and fj==lj and a>2 ):
            break
            
    borders.append([fi,fj])
    print('------------------------a:',a)
    # Shapenning borders pixels 
    for t in range(len(borders)):
        #print(t,borders[t])
        i=borders[t][0]
        j=borders[t][1]
        num[i,j]=100

    file_name=file_name.replace('mask-','') 
    file_name=file_name.replace('.png','.dcm')
    np.save(path+ 'borders/Border'+str(val+2)+'-'+file_name +'.npy', borders)
    print('Border'+str(val+2))
    
    
def Text2Mask(path,name):
    fname=path+'TextImage/'+name

    img = Image.open(fname).convert('L')
    img = np.asarray(img)

    # find connected components
    labeled, nr_objects = ndimage.label(img) 
    
    #print("Number of objects is {}".format(nr_objects))
    # Number of objects is 4 

    plt.imshow(labeled)
    #name=name.replace('.png','.dcm')
    plt.imsave(path+'mask/Text-'+name, labeled)    
    return(nr_objects)

def Text2Image(strText,path,name):
    # Alphabet image
    img = Image.new('RGB', (500, 80), color = (0, 0, 0))

    fnt = ImageFont.truetype('Arial.ttf', 60) #'/Library/Fonts/'
    d = ImageDraw.Draw(img)
    d.text((0,0), strText, font=fnt, fill=(255, 255, 255))

    img.save(path+'TextImage/'+name)

def TextImage2NumpyArray(path):
    files = os.listdir(path+'dicom')
    for i,name in enumerate(files):
        name=files[i]
        if name.find('.dcm')>=0:
            fpath=path+'DICOM/'+name
            ds = dicom.read_file(fpath, force=True)
            print(ds.InstanceNumber)
            strText=' ' + str(ds.InstanceNumber)
            name=name.replace('.dcm','.png')
            Text2Image(strText,path,name)
            nr=Text2Mask(path,name)
            print("Number of objects :",nr)
            for j in range(nr):
                rn=j+1
                print('Boarder number:',rn)
                BorderAlphabet2Numpy(path,name,rn)
'''-------------------------------------------------------------------
    Codify
'''
# Orientation

def file_plane(IOP):
    IOP_round = [round(x) for x in IOP]
    plane = np.cross(IOP_round[0:3], IOP_round[3:6])
    plane = [abs(x) for x in plane]
    if plane[0] == 1:
        return 0    #"Sagittal"
    elif plane[1] == 1:
        return 1    #"Coronal"
    elif plane[2] == 1:
        return 2    #"Transverse"

def newPosition(n,ax,xp_rt,yp_rt,x_rt,y_rt):
    if ax == 0:
        return(xp_rt+n*2*abs(xp_rt)/abs(x_rt))
    else:
        return(yp_rt+n*2*abs(yp_rt)/abs(y_rt))


def DicomRT(path,file_name,region_number):
    file_path=path+'Dicom/'+file_name
    dsorg = pydicom.read_file(file_path, force=True)
    
    dcmfiles = os.listdir(path+'Dicom/')
    
    IOP = dsorg.ImageOrientationPatient
    plane = file_plane(IOP)
    planVal=dsorg.ImagePositionPatient[plane]
    planVal=float(planVal)
    
    xp_rt=dsorg.ImagePositionPatient[0]
    yp_rt=dsorg.ImagePositionPatient[1]

    x_rt=dsorg.Columns
    y_rt=dsorg.Rows
       
    uid1=generate_uid()
    uid2=generate_uid()
    # File meta info data elements
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 182
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    file_meta.MediaStorageSOPInstanceUID = uid1 #'1.2.826.0.1.534147.578.2719282597.2020101685637449'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    file_meta.ImplementationClassUID = '1.2.40.0.13.1.1'
    file_meta.ImplementationVersionName = 'dcm4che-2.0'

    ds = Dataset()

    # Main data elements
    ds = Dataset()
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    ds.SOPInstanceUID =uid1 #'1.2.826.0.1.534147.578.2719282597.2020101685637449' 
    ds.StudyDate =dsorg.StudyDate #'20450916'
    ds.StudyTime =dsorg.StudyTime # '000000'
    ds.AccessionNumber = ''
    ds.Modality = 'RTSTRUCT'
    ds.Manufacturer =dsorg.Manufacturer # 'SIEMENS'
    ds.ReferringPhysicianName = ''
    ds.OperatorsName = ''
    ds.ManufacturerModelName = dsorg.ManufacturerModelName # SOMATOM Definition Edge'
    ds.PatientName = dsorg.PatientName # 'Covid7175'
    ds.PatientID = dsorg.PatientID # 'Covid7175'
    ds.PatientBirthDate = dsorg.PatientBirthDate # '19300101'
    ds.PatientSex = dsorg.PatientSex # 'F'
    ds.SoftwareVersions = dsorg.SoftwareVersions # 'syngo CT VA48A'
    ds.StudyInstanceUID = dsorg.StudyInstanceUID #'1.2.826.0.1.3680043.9.3218.1.1.302475.1985.1592890895061.53221.0' # dsOrg.StudyInstanceUID
    ds.SeriesInstanceUID = uid2 #'1.2.826.0.1.534147.578.2719282597.2020101685637450' 
    ds.StudyID = ''
    ds.SeriesNumber = None
    ds.FrameOfReferenceUID = dsorg.FrameOfReferenceUID #'1.2.826.0.1.3680043.9.3218.1.1.302475.1985.1592890895061.53224.0' # dsOrg.FrameOfReferenceUID
    ds.PositionReferenceIndicator = ''
    ds.StructureSetLabel = 'AIM_Multi3_' + str(dsorg.InstanceNumber) +'_'+ str(region_number) #Scaling04
    ds.StructureSetDate ='20201116'
    ds.StructureSetTime ='085637'

    # Referenced Frame of Reference Sequence
    refd_frame_of_ref_sequence = Sequence()
    ds.ReferencedFrameOfReferenceSequence = refd_frame_of_ref_sequence

    # Referenced Frame of Reference Sequence: Referenced Frame of Reference 1
    refd_frame_of_ref1 = Dataset()
    refd_frame_of_ref1.FrameOfReferenceUID =dsorg.FrameOfReferenceUID # '1.2.826.0.1.3680043.9.3218.1.1.302475.1985.1592890895061.53224.0' 

    # RT Referenced Study Sequence
    rt_refd_study_sequence = Sequence()
    refd_frame_of_ref1.RTReferencedStudySequence = rt_refd_study_sequence

    # RT Referenced Study Sequence: RT Referenced Study 1
    rt_refd_study1 = Dataset()
    rt_refd_study1.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.1'
    rt_refd_study1.ReferencedSOPInstanceUID = dsorg.StudyInstanceUID #'1.2.826.0.1.3680043.9.3218.1.1.302475.1985.1592890895061.53221.0' # 

    # RT Referenced Series Sequence
    rt_refd_series_sequence = Sequence()
    rt_refd_study1.RTReferencedSeriesSequence = rt_refd_series_sequence

    # RT Referenced Series Sequence: RT Referenced Series 1
    rt_refd_series1 = Dataset()
    rt_refd_series1.SeriesInstanceUID =dsorg.SeriesInstanceUID #'1.2.826.0.1.3680043.9.3218.1.1.302475.1985.1592890895061.53222.0'

    # Contour Image Sequence
    contour_image_sequence = Sequence()
    rt_refd_series1.ContourImageSequence = contour_image_sequence

    # Contour Image Sequence: Contour Image 1 ********************************
    i=0
    contour_image=[]    
    for dcmname in dcmfiles:
        if '.dcm' in dcmname:   
            dsorg = pydicom.read_file(path+'Dicom/'+dcmname, force=True)
            contour_image.append(Dataset())
            contour_image[i] = Dataset()
            contour_image[i].ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            contour_image[i].ReferencedSOPInstanceUID = dsorg.SOPInstanceUID #'1.2.826.0.1.3680043.9.3218.1.1.302475.1985.1592890895061.53223.0' 
            contour_image[i].ReferencedFrameNumber = "1"
            contour_image_sequence.append(contour_image[i])            
            
            i=i+1
                    
    # contour_image1 = Dataset()
    # contour_image1.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    # contour_image1.ReferencedSOPInstanceUID = dsorg.SOPInstanceUID #'1.2.826.0.1.3680043.9.3218.1.1.302475.1985.1592890895061.53223.0' 
    # contour_image1.ReferencedFrameNumber = "1"
    # contour_image_sequence.append(contour_image1)
     
    
    rt_refd_series_sequence.append(rt_refd_series1)
    rt_refd_study_sequence.append(rt_refd_study1)
    refd_frame_of_ref_sequence.append(refd_frame_of_ref1)


    # Structure Set ROI Sequence
    structure_set_roi_sequence = Sequence()
    ds.StructureSetROISequence = structure_set_roi_sequence

    # Structure Set ROI Sequence: Structure Set ROI 1
    structure_set_roi1 = Dataset()
    structure_set_roi1.ROINumber = "1"
    structure_set_roi1.ReferencedFrameOfReferenceUID = dsorg.FrameOfReferenceUID #'1.2.826.0.1.3680043.9.3218.1.1.302475.1985.1592890895061.53224.0' # 
    structure_set_roi1.ROIName = 'TestScale'
    structure_set_roi1.ROIGenerationAlgorithm = ''
    structure_set_roi_sequence.append(structure_set_roi1)


    # ROI Contour Sequence
    roi_contour_sequence = Sequence()
    ds.ROIContourSequence = roi_contour_sequence

    # ROI Contour Sequence: ROI Contour 1
    roi_contour1 = Dataset()

    # Contour Sequence
    contour_sequence = Sequence()
    roi_contour1.ContourSequence = contour_sequence

    # Contour Sequence: Contour 1
    contour=[]
    #dcmfiles = os.listdir(path+'Dicom/') came to beginig of the function
    i=0
    for dcmname in dcmfiles:
        #print(dcmname)
        if '.dcm' in dcmname:       
            pnyfiles = os.listdir(path+'borders/')
            for pnyname in pnyfiles:
                if dcmname in pnyname:
                    #print(pnyname)
                    dsorg = pydicom.read_file(path+'Dicom/'+dcmname, force=True)
                
                    IOP = dsorg.ImageOrientationPatient
                    plane = file_plane(IOP)
                    planVal=dsorg.ImagePositionPatient[plane]
                    planVal=float(planVal)
                    
                    xp_rt=dsorg.ImagePositionPatient[0]
                    yp_rt=dsorg.ImagePositionPatient[1]
                
                    x_rt=dsorg.Columns
                    y_rt=dsorg.Rows            
                    # Put Contoure pixel cordination Inside file
                    with open(path+'Borders/'+pnyname, 'rb') as f:
                        num = np.load(f)
                    print(pnyname)  
                    print(planVal)
                    borders=[]
                    for t in range(len(num)):
                        #print(t,num[t])    
                        if plane == 0:  #"Sagittal"
                            x=planVal
                            y=newPosition(num[t][1],0,xp_rt,yp_rt,x_rt,y_rt)
                            z=newPosition(num[t][0],1,xp_rt,yp_rt,x_rt,y_rt)
                        elif plane == 1:  #"Coronal"
                            x=newPosition(num[t][1],0,xp_rt,yp_rt,x_rt,y_rt)
                            y=planVal
                            z=newPosition(num[t][0],1,xp_rt,yp_rt,x_rt,y_rt)
                        elif plane == 2:#  "Transverse"
                            x=newPosition(num[t][1],0,xp_rt,yp_rt,x_rt,y_rt)
                            y=newPosition(num[t][0],1,xp_rt,yp_rt,x_rt,y_rt)
                            z=planVal
                        borders.extend([x,y,z])
        
                    print(i)
                    contour.append(Dataset())
                    contour[i] = Dataset()
                    
                    # Contour Image Sequence
                    contour_image_sequence = Sequence()
                    contour[i].ContourImageSequence = contour_image_sequence
            
                    # Contour Image Sequence: Contour Image 1
                    contour_image1 = Dataset()
                    contour_image1.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
                    contour_image1.ReferencedSOPInstanceUID =dsorg.SOPInstanceUID #'1.2.826.0.1.3680043.9.3218.1.1.302475.1985.1592890895061.53223.0' 
                    contour_image1.ReferencedFrameNumber = "1"
                    contour_image_sequence.append(contour_image1)
                
                    contour[i].ContourGeometricType = 'CLOSED_PLANAR'
                    contour[i].NumberOfContourPoints = len(borders)/3#"4"
                    contour[i].ContourNumber = "1"
                    contour[i].ContourData =borders # [-276.91503267973, -162.50000000000, 516.398692810457, 270.222222222222, -162.50000000000, 514.725490196078, 271.895424836601, -162.50000000000, -177.98039215686, -271.89542483660, -162.50000000000, -176.30718954248]
                    contour_sequence.append(contour[i])
                    i=i+1

    roi_contour1.ReferencedROINumber = "1"
    roi_contour_sequence.append(roi_contour1)


    # RT ROI Observations Sequence
    rtroi_observations_sequence = Sequence()
    ds.RTROIObservationsSequence = rtroi_observations_sequence

    # RT ROI Observations Sequence: RT ROI Observations 1
    rtroi_observations1 = Dataset()
    rtroi_observations1.ObservationNumber = "1"
    rtroi_observations1.ReferencedROINumber = "1"
    rtroi_observations1.RTROIInterpretedType = ''
    rtroi_observations1.ROIInterpreter = ''
    rtroi_observations_sequence.append(rtroi_observations1)

    ds.file_meta = file_meta
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    ds.save_as(path+'RTSTRUCT/rt'+str(region_number)+'-'+file_name, write_like_original=False)
