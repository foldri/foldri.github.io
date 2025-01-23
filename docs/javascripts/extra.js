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