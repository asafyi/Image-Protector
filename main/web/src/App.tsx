
import './index.css'


import  {Container}  from './components/container'

import { DndProvider } from 'react-dnd'
import { HTML5Backend } from 'react-dnd-html5-backend'


function App() {
  //const [count, setCount] = useState(0)

  return (
    <div >
      <header className='app-header'>
    
      <p>Welcome to image protector 5000!</p>
      
      </header>
      
      
     
      
      <DndProvider backend={HTML5Backend}>
					<Container />
      
				</DndProvider>
      <p className="centerText">Here you can upload an image of your face and we'll generate the same image, protected with our defence methods <br/> </p>
      

    </div>
  )
}

export default App