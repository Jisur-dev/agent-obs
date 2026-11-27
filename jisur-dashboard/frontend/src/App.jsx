import { useState } from 'react'
import Aside from './componets/aside.jsx' 


function App() {
  const [activeId,setActiveId]=useState('dashboard');
  return( 
    <div>
      <Aside activeId={activeId} onNavigate={setActiveId} />
    </div>
  )
}

export default App
