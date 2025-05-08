using Microsoft.AspNetCore.Mvc;
using BackEnd;
using BackEnd.DBContext;
using BackEnd.Models;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using System;
using Microsoft.AspNetCore.Http;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;

namespace BackEnd.Controllers
{
	[ApiController]
	[Route("api/[controller]")]
	public class LearningController : ControllerBase
	{
		private readonly ApplicationDbContext _context;

		public LearningController(ApplicationDbContext context)
		{
			_context = context;
		}
		// Эндпоинт для получения категорий
		[HttpGet("Category")]
		public async Task<ActionResult<IEnumerable<Category>>> GetCategories()
		{
			var category = await _context.Category.ToListAsync();
			if (category == null || category.Count == 0)
			{
				return NotFound("Категории не найдены.");
			}
			return Ok(category);
		}
		[HttpGet("TestCategory")]
		public async Task<ActionResult<IEnumerable<TestCategory>>> GetTestCategories()
		{
			var categories = await _context.TestCategory.ToListAsync();
			if (categories == null || categories.Count == 0)
			{
				return NotFound("Категории тестов не найдены.");
			}
			return Ok(categories);
		}
		// Эндпоинт для получения уровней сложности
		[HttpGet("DifficultyLevel")]
		public async Task<ActionResult<IEnumerable<DifficultyLevel>>> GetDifficultyLevel()
		{
			var levels = await _context.DifficultyLevel.ToListAsync();
			if (levels == null || levels.Count == 0)
			{
				return NotFound("Уровни сложности не найдены.");
			}
			return Ok(levels);
		}

		// Эндпоинт для получения предложений с фильтрацией по категории и уровню сложности
		[HttpGet("MissingWord")]
		public async Task<ActionResult<IEnumerable<MissingWord>>> GetMissingWord([FromQuery] int? categoryId, [FromQuery] int? difficultyLevelId)
		{
			var query = _context.MissingWord.AsQueryable();

			// Фильтрация по категории
			if (categoryId.HasValue)
			{
				query = query.Where(m => m.CategoryId == categoryId.Value);
			}

			// Фильтрация по уровню сложности
			if (difficultyLevelId.HasValue)
			{
				query = query.Where(m => m.DifficultyLevelId == difficultyLevelId.Value);
			}

			var missingWords = await query.ToListAsync();

			// Если предложений нет, возвращаем 404 с сообщением
			if (missingWords == null || missingWords.Count == 0)
			{
				return NotFound(new { message = "Предложения не найдены для выбранной категории и уровня сложности." });
			}

			return Ok(missingWords);
		}

		[HttpPost("MissingWord")]
		public async Task<ActionResult<MissingWord>> PostMissingWord(MissingWord missingWord)
		{
			if (missingWord == null)
			{
				return BadRequest("Неверные данные.");
			}

			_context.MissingWord.Add(missingWord);
			await _context.SaveChangesAsync();

			return CreatedAtAction(nameof(GetMissingWord), new { id = missingWord.Id }, missingWord);
		}
		// GET: api/Learning/Flashcards
		[HttpGet("Flashcards")]
		public async Task<ActionResult<IEnumerable<Flashcard>>> GetFlashcards()
		{

			// Получаем все карточки из базы данных
			var flashcards = await _context.Flashcards.ToListAsync();

			// Если нет карточек в базе, возвращаем 404 Not Found
			if (flashcards == null || flashcards.Count == 0)
			{
				return NotFound("Карточки не найдены.");
			}

			return Ok(flashcards);  // Возвращаем список карточек
		}

		// POST: api/Learning/Flashcards
		[HttpPost("Flashcards")]
		public async Task<ActionResult<Flashcard>> PostFlashcard(Flashcard flashcard)
		{
			if (flashcard == null)
			{
				return BadRequest("Неверные данные.");
			}

			// Добавляем новую карточку в базу данных
			_context.Flashcards.Add(flashcard);
			await _context.SaveChangesAsync();

			return CreatedAtAction(nameof(GetFlashcards), new { id = flashcard.Id }, flashcard);
		}

		// ✅ GET: api/learning/test — Получить все тесты с категориями и уровнем сложности
		[HttpGet("Test")]
		public async Task<ActionResult> GetTests()
		{
			var tests = await _context.Test
				.Include(t => t.TestCategory)        // Включаем связанную категорию
				.Include(t => t.DifficultyLevel)     // Включаем уровень сложности
				.Select(t => new
				{
					t.Id,
					t.Title,
					t.Description,
					t.TestCategoryId,
					t.DifficultyLevelId,
					Category = t.TestCategory.Name,     // Плоское поле
					Difficulty = t.DifficultyLevel.Name // Плоское поле
				})
				.ToListAsync();

			if (tests == null || tests.Count == 0)
			{
				return NotFound("Тесты не найдены.");
			}

			return Ok(tests);
		}




		// ✅ GET: api/learning/test/{id} — Получить тест с вопросами и вариантами ответов
		[HttpGet("TestFiltered")]
		public async Task<ActionResult> GetFilteredTests(
	[FromQuery] int? testCategoryId,
	[FromQuery] int? difficultyLevelId,
	[FromQuery] int? id)
		{
			var query = _context.Test
				.Include(t => t.TestCategory)
				.Include(t => t.DifficultyLevel)
				.Include(t => t.Questions)
					.ThenInclude(q => q.AnswerOptions)
				.AsQueryable();

			// ✅ Запрос конкретного теста с вопросами и вариантами ответов
			if (id.HasValue)
			{
				var test = await query
					.Where(t => t.Id == id.Value)
					.Select(t => new
					{
						t.Id,
						t.Title,
						t.Description,
						Category = t.TestCategory.Name,
						Difficulty = t.DifficultyLevel.Name,
						Questions = t.Questions.Select(q => new
						{
							q.Id,
							QuestionText = q.QuestionText,
							QuestionType = q.QuestionType,
							AnswerOptions = q.AnswerOptions.Select(a => new
							{
								a.Id,
								AnswerText = a.AnswerText,
								IsCorrect = a.IsCorrect
							}).ToList()
						}).ToList()
					})
					.FirstOrDefaultAsync();

				if (test == null) return NotFound("Тест не найден");

				return Ok(test);
			}

			// ✅ Запрос всех тестов без вопросов (для списка)
			if (testCategoryId.HasValue)
				query = query.Where(t => t.TestCategoryId == testCategoryId.Value);

			if (difficultyLevelId.HasValue)
				query = query.Where(t => t.DifficultyLevelId == difficultyLevelId.Value);

			var tests = await query
				.Select(t => new
				{
					t.Id,
					t.Title,
					t.Description,
					Category = t.TestCategory.Name,
					Difficulty = t.DifficultyLevel.Name
				})
				.ToListAsync();

			return tests.Count == 0 ? NotFound("Тесты не найдены") : Ok(tests);
		}



		[HttpPost("Pronunciation")]
		public async Task<IActionResult> PostPronunciation([FromForm] PronunciationRequest request)
		{
			if (request?.AudioData == null || string.IsNullOrEmpty(request.Word))
			{
				return BadRequest("AudioData and Word are required");
			}

			if (request.AudioData.ContentType != "audio/wav" &&
				request.AudioData.ContentType != "audio/x-wav" &&
				request.AudioData.ContentType != "audio/mpeg" &&
				request.AudioData.ContentType != "audio/wave")
			{
				return BadRequest("Only WAV or MP3 files are supported");
			}

			using var memoryStream = new MemoryStream();
			await request.AudioData.CopyToAsync(memoryStream);
			var audioBytes = memoryStream.ToArray();

			// Создание HTTP  для FastAPI
			using var client = new HttpClient();
			using var content = new MultipartFormDataContent();

			content.Add(new ByteArrayContent(audioBytes), "file", "audio.wav");
			content.Add(new StringContent(request.Word), "word");//?

			try
			{
				var response = await client.PostAsync("http://localhost:8000/predict", content);

				if (!response.IsSuccessStatusCode)
				{
					var error = await response.Content.ReadAsStringAsync();
					return StatusCode((int)response.StatusCode, error);
				}

				var result = await response.Content.ReadFromJsonAsync<dynamic>();

				return Ok(new
				{
					Word = request.Word,
					Score = result?.score,  
					AudioSize = audioBytes.Length
				});
			}
			catch (Exception ex)
			{
				return StatusCode(500, new { error = "Error calling FastAPI", details = ex.Message });
			}
		}


		public class PronunciationRequest
		{
			public string Word { get; set; }  
			public IFormFile AudioData { get; set; }  
		}


		//[HttpGet("Words")]
		//public async Task<IActionResult> GetWords()
		//{
		//	var words = await _context.PronunciationRequests
		//							  .Select(w => w.Word)
		//							  .Distinct()
		//							  .ToListAsync();

		//	return Ok(words);
		//}


	}

}
