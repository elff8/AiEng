using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using BackEnd.Models; // Добавьте эту директиву

namespace BackEnd.Models
{
    public class UserProfiles
	{
        public int Id { get; set; }
        public int UserId { get; set; } // ID пользователя
        public int CorrectAnswersCount { get; set; } // Количество правильных ответов для этой карточки
        public bool IsLearned { get; set; } // Статус, выучено ли слово
    }

}
