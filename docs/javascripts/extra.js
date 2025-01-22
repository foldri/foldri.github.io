function hover1(element) {
    element.setAttribute('src', '../about/FrederikBennhoff_Photo.jpg');
  }
  
function unhover1(element) {
  element.setAttribute('src', '../about/FrederikBennhoff_Photo_BW.png');
}

function hover2(element) {
  element.setAttribute('src', '../about/IgliBajo_Photo.jpeg');
}

function unhover2(element) {
  element.setAttribute('src', '../about/IgliBajo_Photo_BW.jpg');
}

function changeRef() {
  const logo = document.querySelector(".md-header__button.md-logo");
  if (logo) {
      logo.href = "../../../../about/";
  }
}

function changeRefnav() {
  const logo = document.querySelector(".md-nav__title a");
  if (logo) {
      logo.href = "../../../../about/";
  }
}


document.addEventListener("DOMContentLoaded", changeRef);
document.addEventListener("DOMContentLoaded", changeRefnav);