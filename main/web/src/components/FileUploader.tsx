import React  from "react";

import UploadFileIcon from '@mui/icons-material/UploadFile';




interface FileUploaderProps {
  onFileUpload: (files: FileList | null) => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileUpload }) => {
 

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      
      onFileUpload(event.target.files);
    }
  };

  return (
    <div  className="lowerBoxStyle">
      <br/><br/>
      <p className="orSize">Or:</p>
      <br/><br/>
      <input
        type="file"
        id="file-uploader"
        style={{ display: 'none' }}
        multiple
        onChange={handleFileChange}
      />
      <label htmlFor="file-uploader">
        <div className="uploadButton ">
      upload image<br/><br/>
      <UploadFileIcon  fontSize="large"/>
        </div>
      </label>
    </div>
  );
};
export default FileUploader;