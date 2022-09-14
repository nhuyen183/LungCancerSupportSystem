// // modal open
// const notSureBtn = document.querySelector(".js-not-sure");
// const modal = document.querySelector(".js-modals");
// const closeBtns = document.querySelectorAll(".js-close");
// const modalBody = document.querySelector(".js-modal__body");

// function nutritionClose() {
//   modal.classList.remove("open");
// }

// function nutritionModal() {
//   modal.classList.add("open");
// }

// notSureBtn.addEventListener("click", nutritionModal);
// for (var closeBtn of closeBtns) {
//   closeBtn.addEventListener("click", nutritionClose);
// }

// modal.addEventListener("click", nutritionClose);

// modalBody.addEventListener("click", function (event) {
//   event.stopPropagation();
// });

// const result = document.querySelector(".result");
// const apply_result = document.querySelector(".apply-result");
// const calculate_btn = document.querySelector(".condensed-form");

// calculate_btn.addEventListener("submit", function(e) {
//   result.classList.add("open");
//   apply_result.classList.add("open");
//   e.preventDefault();
// });

// category active change

const category_list = document.getElementById("js-category-list");
const categoryItems = category_list.getElementsByTagName("button");
// const modal__currentDietType = document.querySelector(".js-modal__current-diet-type span");
// var dietType;

for (var categoryItem of categoryItems) {
  categoryItem.addEventListener("click", function () {
    var cur = document.querySelector("#js-category-list .active");
    cur.className = cur.className.replace("active", "");
    this.className += "active";
    // dietType = document.querySelector(".js-category-item.active span");
    // if (this.className === "js-category-item active") {
    // modal__currentDietType.innerHTML = dietType.innerHTML;
    // }
  });
}

//radio checked change
const goal_radio = document.querySelector("#goal");
const goal_Items = goal_radio.getElementsByTagName("label");

goal_radio.addEventListener(
  "change",
  function (e) {
    let trg = e.target;
    console.log(trg);
    let trg_par = trg.parentElement;
    console.log(trg_par);

    if (
      trg.type === "radio" &&
      trg_par &&
      trg_par.tagName.toLowerCase() === "label"
    ) {
      let prior = goal_radio.querySelector(
        'label.checked input[name="' + trg.name + '"]'
      );
      console.log(prior);
      if (prior) {
        prior.parentElement.classList.remove("checked");
      }
      trg_par.classList.add("checked");
    }
  },
  false
);

const gender_radio = document.querySelector("#gender-select");
const gender_Items = gender_radio.getElementsByTagName("label");

gender_radio.addEventListener(
  "change",
  function (e) {
    let trg = e.target;
    console.log(trg);
    let trg_par = trg.parentElement;
    console.log(trg_par);

    if (
      trg.type === "radio" &&
      trg_par &&
      trg_par.tagName.toLowerCase() === "label"
    ) {
      let prior = gender_radio.querySelector(
        'label.checked input[name="' + trg.name + '"]'
      );
      console.log(prior);
      if (prior) {
        prior.parentElement.classList.remove("checked");
      }
      trg_par.classList.add("checked");
    }
  },
  false
);

function toggleBoxVisibility() {
  if (document.getElementById("nutri-info").checked == true) {
    document.getElementById("box").style.visibility = "visible";
  } else {
    document.getElementById("box").style.visibility = "hidden";
  }
}

let weight_change_ratio = document.querySelector(
  "#weight-change-rate-input"
).value;
